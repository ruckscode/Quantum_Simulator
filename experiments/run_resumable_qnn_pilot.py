"""One bounded, source-specific QNN pilot with explicit epoch-boundary resume."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
TASK_CONFIGS = {
    "bugs4q_joint_source_state": {
        "contract": "bugs4q_joint_source_state",
        "processed": "data/processed/bugs4q_joint/source_state",
        "output": "reports/experiments/resumable_qnn_pilot_bugs4q_joint_20261029",
        "rows": 124, "groups": 62,
        "target_semantics": "Buggy versus fixed source-state classification on paired Bugs4Q core and NA examples, grouped by issue; not hardware diagnosis.",
        "group_column": "bug_id", "group_unit": "source issue group",
    },
    "bugs4q_core_only_transfer_source_restricted_v2": {
        "contract": "bugs4q_core_only_transfer",
        "processed": "data/processed/bugs4q_joint/core_only_transfer",
        "output": "reports/experiments/resumable_qnn_pilot_bugs4q_core_transfer_v2_20261029",
        "rows": 124, "groups": 62,
        "target_semantics": "Train each grouped fold on Bugs4Q_core only; evaluate on its held-out fold across Bugs4Q_core and Bugs4Q_NA_additional, reporting the NA transfer subset separately. Grouped buggy/fixed source-state prediction, not hardware diagnosis.",
        "group_column": "bug_id", "group_unit": "source issue group",
        "training_source": "Bugs4Q_core", "transfer_target_source": "Bugs4Q_NA_additional",
    },
    "bugs4q_na_only_transfer_source_restricted_v2": {
        "contract": "bugs4q_na_only_transfer",
        "processed": "data/processed/bugs4q_joint/na_only_transfer",
        "output": "reports/experiments/resumable_qnn_pilot_bugs4q_na_transfer_v2_20261029",
        "rows": 124, "groups": 62,
        "target_semantics": "Train each grouped fold on Bugs4Q_NA_additional only; evaluate on its held-out fold across Bugs4Q_NA_additional and Bugs4Q_core, reporting the core transfer subset separately. Grouped buggy/fixed source-state prediction, not hardware diagnosis.",
        "group_column": "bug_id", "group_unit": "source issue group",
        "training_source": "Bugs4Q_NA_additional", "transfer_target_source": "Bugs4Q_core",
    },
    "bugs4q_na_source_state": {
        "processed": "data/processed/bugs4q_na/source_state",
        "output": "reports/experiments/resumable_qnn_pilot_bugs4q_na_20261029",
        "rows": 62, "groups": 31,
        "target_semantics": "Buggy versus fixed source-state classification grouped by issue; not hardware diagnosis.",
        "group_column": "bug_id",
    },
    "bugs4q_core_source_state": {
        "processed": "data/processed/bugs4q_core/source_state",
        "output": "reports/experiments/resumable_qnn_pilot_bugs4q_core_20261029",
        "rows": 62, "groups": 31,
        "target_semantics": "Buggy versus fixed source-state classification grouped by issue; not hardware diagnosis.",
        "group_column": "bug_id",
    },
    "bugs4q_robust_execution": {
        "processed": "data/processed/bugs4q_robust/harness_execution_outcome_v2",
        "output": "reports/experiments/resumable_qnn_pilot_bugs4q_robust_20261029",
        "rows": 2220, "groups": 37,
        "target_semantics": "Observed test-harness Pass versus Fail for a program snapshot; repeated executions grouped by issue. Not a universal bug label or hardware diagnosis.",
        "group_column": "bug_id",
    },
    "qolumbina_ideal_seeded_mutant_detection": {
        "contract": "qolumbina_ideal_detection",
        "processed": "data/processed/qolumbina_ideal/seeded_mutant_detection_ideal",
        "output": "reports/experiments/resumable_qnn_pilot_qolumbina_ideal_20261029",
        "rows": 182, "groups": 26,
        "target_semantics": "Predict whether DOSS detects a seeded mutant in a majority of available ideal-backend report seeds; not a ground-truth repair label or hardware diagnosis.",
        "group_column": "bug_id", "group_unit": "seeded-mutant source program group",
    },
    "qolumbina_multibackend_seeded_mutant_detection": {
        "contract": "qolumbina_multibackend_detection",
        "processed": "data/processed/qolumbina/seeded_mutant_detection_multibackend",
        "output": "reports/experiments/resumable_qnn_pilot_qolumbina_multibackend_20261029",
        "rows": 546, "groups": 26,
        "target_semantics": "Predict whether DOSS detects a seeded Qolumbina mutant in a majority of available report seeds across ideal, fake_algiers, and fake_brooklyn. This empirical mutation-outcome target is not ground-truth repair or hardware diagnosis.",
        "group_column": "bug_id", "group_unit": "original Qolumbina program across all variants and backends",
        "feature_fields": ["code", "backend"],
    },
    "qolumbina_multibackend_seeded_mutant_detection_balanced": {
        "contract": "qolumbina_multibackend_detection",
        "processed": "data/processed/qolumbina/seeded_mutant_detection_multibackend",
        "output": "reports/experiments/resumable_qnn_pilot_qolumbina_multibackend_balanced_20261029",
        "rows": 546, "groups": 26,
        "target_semantics": "Exploratory fold-local class-balanced loss variant for empirical Qolumbina seeded-mutant detection across ideal, fake_algiers, and fake_brooklyn; not ground-truth repair or hardware diagnosis.",
        "group_column": "bug_id", "group_unit": "original Qolumbina program across all variants and backends",
        "feature_fields": ["code", "backend"], "class_weighting": "balanced",
    },
    "lintq_warning_triage_context_v2": {
        "processed": "data/processed/lintq/warning_triage_context_v2",
        "output": "reports/experiments/resumable_qnn_pilot_lintq_context_v2_20261029",
        "rows": 290, "groups": 217,
        "target_semantics": "Manual LintQ TP versus FP triage from rule, warning, and source code context; not hardware diagnosis.",
        "group_column": "model_group_id",
    },
    "lintq_warning_triage_context_v2_balanced": {
        "contract": "lintq_warning_triage_context_v2",
        "processed": "data/processed/lintq/warning_triage_context_v2",
        "output": "reports/experiments/resumable_qnn_pilot_lintq_context_v2_balanced_20261029",
        "rows": 290, "groups": 217,
        "target_semantics": "Manual LintQ TP versus FP triage from rule, warning, and source code context, with fold-local balanced loss; exploratory and not hardware diagnosis.",
        "group_column": "model_group_id", "class_weighting": "balanced",
    },
    "quantum_platform_comment_v2_balanced": {
        "contract": "quantum_platform_comment_v2",
        "processed": "data/processed/quantum_platform_bugs/comment_only_classification_v2",
        "output": "reports/experiments/resumable_qnn_pilot_quantum_platform_v2_balanced_20261029",
        "rows": 319, "groups": 304,
        "target_semantics": "Curated quantum-platform issue-comment annotations, grouped by issue and exact comment; labels are not hardware fault labels.",
        "group_column": "comment_component_id", "class_weighting": "balanced",
    },
    "qbugs_qasm_v2_source_state": {
        "contract": "qbugs_correct_buggy_circuit_state_qasm_v2",
        "processed": "data/processed/qbugs_hornbro/correct_buggy_circuit_state_qasm_v2",
        "output": "reports/experiments/resumable_qnn_pilot_qbugs_qasm_v2_20261029",
        "rows": 1496, "groups": 748,
        "target_semantics": "Upstream paired correct versus buggy QBugs circuit outcome from full OpenQASM 2 text; two contradictory exact-QASM source pairs are rejected. Not hardware diagnosis.",
        "group_column": "bug_id", "group_unit": "complete source pair connected by exact QASM", "class_weighting": "balanced",
    },
    "mute_alg_hornbro_qasm_v2": {
        "contract": "mute_alg_compatible_subset_qasm_v2_qiskit_compat_v2",
        "processed": "data/processed/mute_alg/compatible_subset_qasm_v2_qiskit_compat_v2",
        "output": "reports/experiments/resumable_qnn_pilot_mute_alg_qasm_v2_20261029",
        "rows": 9140, "groups": 812,
        "target_semantics": "Upstream Mute-Alg original (0) versus synthetically buggy (1) circuits from HornBro; paired records and exact OpenQASM repeats are grouped. Historical source repair outcome, not hardware diagnosis.",
        "group_column": "bug_id", "group_unit": "source pair connected by exact OpenQASM recurrence",
    },
}
TASK = "bugs4q_na_source_state"
PROCESSED = ROOT / TASK_CONFIGS[TASK]["processed"]
CONTRACT = ROOT / f"experiments/source_task_contracts/{TASK}.json"
OUT = ROOT / TASK_CONFIGS[TASK]["output"]
SEED = 28102026
FINAL_EPOCHS = 4
BATCH_SIZE = 64
MODEL_CONFIG = {"name": "amplitude_3q_depth1_lr004", "qubits": 3, "features": 8,
                "encoding": "amplitude", "depth": 1, "reupload": False,
                "learning_rate": 0.04}
RAM_FLOOR_MIB = 500


def configure_task(task: str) -> None:
    """Select a registered source contract and isolated output directory."""
    if task not in TASK_CONFIGS:
        raise ValueError(f"Unregistered pilot task: {task}")
    global TASK, PROCESSED, CONTRACT, OUT
    TASK = task
    PROCESSED = ROOT / TASK_CONFIGS[TASK]["processed"]
    CONTRACT = ROOT / f"experiments/source_task_contracts/{TASK_CONFIGS[TASK].get('contract', TASK)}.json"
    OUT = ROOT / TASK_CONFIGS[TASK]["output"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, default=str)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def atomic_csv(path: Path, rows: list[dict], fields: list[str]) -> str:
    import io
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    data = buffer.getvalue().encode("utf-8")
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)
    return hashlib.sha256(data).hexdigest()


def read_task() -> tuple[list[dict], dict, dict]:
    processed_manifest_path = PROCESSED / "manifest.json"
    split_path = PROCESSED / "manifests/group_folds.csv"
    processed = json.loads(processed_manifest_path.read_text(encoding="utf-8"))
    task_spec = TASK_CONFIGS[TASK]
    if not processed.get("validation_pass") or processed["counts"].get("accepted") != task_spec["rows"]:
        raise ValueError(f"{TASK} processed manifest is not validated or has an unexpected row count")
    if sha256(ROOT / processed["input_path"]) != processed["input_sha256"]:
        raise ValueError("Task input source hash changed")
    rows = []
    for part in processed["partitions"]:
        path = ROOT / part["path"]
        if sha256(path) != part["sha256"]:
            raise ValueError(f"Processed partition checksum mismatch: {path}")
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows.extend(csv.DictReader(stream))
    with split_path.open(encoding="utf-8-sig", newline="") as stream:
        split_rows = list(csv.DictReader(stream))
    if sha256(split_path) != processed["split_manifest"]["sha256"]:
        raise ValueError("Group split manifest checksum mismatch")
    assignment = {row["record_id"]: (int(row["fold"]), row["group_id"]) for row in split_rows}
    if len(assignment) != len(split_rows) or set(assignment) != {r["record_id"] for r in rows}:
        raise ValueError("Data/split IDs are missing, duplicated, or inconsistent")
    labels_by_group: dict[str, set[int]] = defaultdict(set)
    group_column = task_spec["group_column"]
    source_hashes = set()
    for row in rows:
        y = int(row["label"])
        if y not in (0, 1) or not row.get("code", "").strip():
            raise ValueError(f"Invalid target or missing code: {row.get('record_id')}")
        fold, group = assignment[row["record_id"]]
        source_group = str(row[group_column])
        if group != source_group:
            raise ValueError(f"Group assignment mismatch for {row['record_id']}")
        labels_by_group[source_group].add(y)
        source_hashes.add(row.get("raw_sha256", ""))
    if len(labels_by_group) != task_spec["groups"] or set(y for labels in labels_by_group.values() for y in labels) != {0, 1}:
        raise ValueError("Unexpected issue-group count or missing binary target class")
    folds_by_group: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        fold, group = assignment[row["record_id"]]
        folds_by_group[group].add(fold)
    if any(len(value) != 1 for value in folds_by_group.values()):
        raise ValueError("An issue group spans multiple folds")
    return rows, assignment, {"processed_manifest": processed_manifest_path,
        "processed_manifest_sha256": sha256(processed_manifest_path),
        "source_path": ROOT / processed["input_path"], "source_sha256": processed["input_sha256"],
        "fold_path": split_path, "fold_sha256": sha256(split_path),
        "contract_sha256": sha256(CONTRACT), "partition_sha256": [x["sha256"] for x in processed["partitions"]],
        "group_count": len(labels_by_group), "record_count": len(rows),
        "raw_snapshot_hashes": len(source_hashes)}


def metric_dict(y, pred) -> dict:
    from train_resource_light_qnn_tasks import metrics
    return metrics(y, pred)


def training_fold_balanced_sample_weights(y: np.ndarray) -> np.ndarray:
    """Compute balanced weights from a training fold only."""
    y = np.asarray(y, dtype=int)
    counts = np.bincount(y, minlength=2)
    if len(y) == 0 or np.any(counts == 0):
        raise ValueError("Balanced binary weights require both classes in the training fold")
    return (len(y) / (2.0 * counts))[y]


def build_fold_indices(rows: list[dict], assignment: dict[str, tuple[int, str]], fold: int,
                       training_source: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return held-out-fold rows and permitted training rows without group leakage."""
    valid = np.asarray([i for i, row in enumerate(rows)
                        if assignment[row["record_id"]][0] == fold], dtype=int)
    train = np.asarray([i for i, row in enumerate(rows)
                        if assignment[row["record_id"]][0] != fold and
                        (training_source is None or row.get("source") == training_source)], dtype=int)
    if len(train) == 0 or len(valid) == 0:
        raise ValueError(f"Fold {fold} has an empty training or validation partition")
    train_groups = {str(rows[i]["bug_id"]) for i in train}
    valid_groups = {str(rows[i]["bug_id"]) for i in valid}
    if train_groups & valid_groups:
        raise ValueError(f"Group leakage in fold {fold}")
    if training_source and any(rows[i].get("source") != training_source for i in train):
        raise ValueError("Source-restricted fold contains records outside the declared training source")
    return train, valid


def run(stop_after_epochs: int | None, resume: bool) -> dict:
    available_start = round(psutil.virtual_memory().available / 2**20, 1)
    if available_start < RAM_FLOOR_MIB:
        raise MemoryError(f"Pilot requires {RAM_FLOOR_MIB} MiB RAM; only {available_start} MiB is available")
    import sys
    sys.path.insert(0, str(ROOT / "experiments"))
    import train_resource_light_qnn_tasks as q
    q.check_ram()
    q.STEPS, q.BATCH, q.SEED = FINAL_EPOCHS, BATCH_SIZE, SEED
    rows, assignment, sources = read_task()
    ids = [row["record_id"] for row in rows]
    y = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    groups = [row[TASK_CONFIGS[TASK]["group_column"]] for row in rows]
    feature_fields = TASK_CONFIGS[TASK].get("feature_fields", ["code"])
    raw_x = np.asarray([q.features_from_text(" ".join(str(row[field]) for field in feature_fields), dims=8)
                        for row in rows], dtype=np.float64)
    folds = sorted({assignment[rid][0] for rid in ids})
    model_config = dict(MODEL_CONFIG)
    if TASK_CONFIGS[TASK].get("class_weighting"):
        model_config["loss"] = "training-fold class-balanced binary cross entropy"
    config = {"task_id": TASK, "target_semantics": TASK_CONFIGS[TASK]["target_semantics"],
              "model": model_config, "seed": SEED,
              "final_epochs": FINAL_EPOCHS, "batch_size": BATCH_SIZE,
              "optimizer": "Adam", "learning_rate": MODEL_CONFIG["learning_rate"],
              "text_features": f"8-dimensional deterministic BLAKE2 hashed-token counts of {', '.join(feature_fields)}",
              "scaler": "minmax fitted on each fold's training rows only",
              "split": f"reused saved 5-fold assignments grouped by {TASK_CONFIGS[TASK].get('group_unit', 'source task group')}",
              "source_hash": sources["source_sha256"], "processed_manifest_sha256": sources["processed_manifest_sha256"],
              "split_sha256": sources["fold_sha256"], "contract_sha256": sources["contract_sha256"]}
    training_source = TASK_CONFIGS[TASK].get("training_source")
    transfer_target_source = TASK_CONFIGS[TASK].get("transfer_target_source")
    if training_source:
        config["training_source_filter"] = training_source
        config["transfer_target_source"] = transfer_target_source
        observed_sources = {str(row.get("source", "")) for row in rows}
        if training_source not in observed_sources or transfer_target_source not in observed_sources or training_source == transfer_target_source:
            raise ValueError("Registered transfer task source filters do not match the validated source rows")
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    run_manifest_path = OUT / "run_manifest.json"
    prior = None
    if OUT.exists() and any(OUT.iterdir()):
        if not resume:
            raise FileExistsError("Pilot output exists; use --resume to verify and continue its checkpointed run")
        if run_manifest_path.is_file():
            prior = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        else:
            # Recover a process interrupted between a durable epoch checkpoint
            # and the aggregate run-manifest write. Only self-fingerprinted
            # sidecars with matching hashes/config are admitted.
            recovered = []
            for sidecar_path in sorted((OUT / "qnn" / MODEL_CONFIG["name"]).glob("fold_*/checkpoint_manifest.json")):
                sidecar_record = json.loads(sidecar_path.read_text(encoding="utf-8"))
                checkpoint_path = sidecar_path.parent / "checkpoint.npz"
                history_path = sidecar_path.parent / "training_history.json"
                if (sidecar_record.get("run_config_sha256") != config_hash or
                        not checkpoint_path.is_file() or not history_path.is_file() or
                        sha256(checkpoint_path) != sidecar_record.get("checkpoint_sha256") or
                        sha256(history_path) != sidecar_record.get("history_sha256")):
                    raise ValueError(f"Cannot recover stale or damaged checkpoint: {sidecar_path}")
                recovered.append({"fold": int(sidecar_record["fold"]), "epoch": int(sidecar_record["epoch"]),
                                  "checkpoint_sha256": sidecar_record["checkpoint_sha256"],
                                  "history_sha256": sidecar_record["history_sha256"]})
            if not recovered:
                raise FileNotFoundError("Cannot resume: no valid checkpoint receipts or run manifest exist")
            prior = {"status": "recovered_from_epoch_receipts", "config_sha256": config_hash,
                     "completed_folds": recovered, "resume_events": 0}
        if prior.get("config_sha256") != config_hash:
            raise ValueError("Pilot configuration/source/split changed; refusing unsafe resume")
        if prior.get("status") == "completed":
            raise FileExistsError("Pilot already completed; refusing duplicate training")
        for record in prior.get("completed_folds", []):
            if record["fold"] not in folds:
                raise ValueError("Pilot fold manifest no longer matches")
    elif resume:
        raise FileNotFoundError("Cannot resume: no checkpointed pilot manifest exists")
    OUT.mkdir(parents=True, exist_ok=True)

    # Classical control uses exactly the same hashed inputs, train-only scaling,
    # group folds, and development labels as the QNN.
    baseline_path = OUT / "classical_oof_predictions.csv"
    baseline_manifest_path = OUT / "classical_baseline_manifest.json"
    if baseline_manifest_path.exists():
        baseline_manifest = json.loads(baseline_manifest_path.read_text(encoding="utf-8"))
        if baseline_manifest.get("config_sha256") != config_hash or sha256(baseline_path) != baseline_manifest.get("predictions_sha256"):
            raise ValueError("Saved classical baseline is incompatible or corrupt")
        baseline_oof = list(csv.DictReader(baseline_path.open(encoding="utf-8-sig", newline="")))
    else:
        from sklearn.linear_model import LogisticRegression
        baseline_oof = []
        for fold in folds:
            train, valid = build_fold_indices(rows, assignment, fold, training_source)
            xt, xv, scaler = q.minmax_fit_apply(raw_x[train].tolist(), raw_x[valid].tolist())
            model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED + fold)
            model.fit(np.asarray(xt), y[train])
            prob = model.predict_proba(np.asarray(xv))[:, list(model.classes_).index(1)]
            pred = (prob >= 0.5).astype(int)
            for j, index in enumerate(valid):
                baseline_oof.append({"record_id": ids[index], "group_id": groups[index], "fold": fold,
                                     "y_true": int(y[index]), "probability": float(prob[j]), "y_pred": int(pred[j])})
        baseline_oof.sort(key=lambda row: row["record_id"])
        digest = atomic_csv(baseline_path, baseline_oof, ["record_id", "group_id", "fold", "y_true", "probability", "y_pred"])
        atomic_json(baseline_manifest_path, {"status": "complete", "model": "LogisticRegression(class_weight=balanced)",
            "config_sha256": config_hash, "predictions_sha256": digest, "feature_width": 8,
            "features_fit_scope": "same fold-local training-only minmax transform and 8 hashed text features as QNN"})

    # Check QNN checkpoint provenance and continue the exact optimizer/RNG state.
    qnn_folder = OUT / "qnn" / MODEL_CONFIG["name"]
    completed_folds = {int(item["fold"]): item for item in (prior or {}).get("completed_folds", [])}
    qnn_oof = []
    fold_results = []
    invocation_start = time.perf_counter()
    invocation_epochs = stop_after_epochs if stop_after_epochs is not None else FINAL_EPOCHS
    if invocation_epochs < 1 or invocation_epochs > FINAL_EPOCHS:
        raise ValueError("stop-after-epochs must be between 1 and the final epoch count")
    for fold in folds:
        train, valid = build_fold_indices(rows, assignment, fold, training_source)
        xt, xv, scaler = q.minmax_fit_apply(raw_x[train].tolist(), raw_x[valid].tolist())
        xt, xv = np.asarray(xt, dtype=float), np.asarray(xv, dtype=float)
        fold_dir = qnn_folder / f"fold_{fold}"
        checkpoint, history_path = fold_dir / "checkpoint.npz", fold_dir / "training_history.json"
        sidecar = fold_dir / "checkpoint_manifest.json"
        fold_provenance = {"run_config_sha256": config_hash, "fold": fold, "seed": SEED + fold,
                           "train_record_ids_sha256": hashlib.sha256("\n".join(sorted(ids[i] for i in train)).encode()).hexdigest(),
                           "validation_record_ids_sha256": hashlib.sha256("\n".join(sorted(ids[i] for i in valid)).encode()).hexdigest(),
                           "scaler_sha256": hashlib.sha256(json.dumps(scaler, sort_keys=True).encode()).hexdigest()}
        sample_weights = None
        if TASK_CONFIGS[TASK].get("class_weighting") == "balanced":
            sample_weights = training_fold_balanced_sample_weights(y[train])
            fold_provenance["sample_weights_sha256"] = hashlib.sha256(sample_weights.tobytes()).hexdigest()
        if checkpoint.exists():
            if not resume or not sidecar.is_file():
                raise ValueError(f"Checkpoint lacks explicit resume authorization/provenance: {checkpoint}")
            saved_meta = json.loads(sidecar.read_text(encoding="utf-8"))
            if (saved_meta.get("run_config_sha256") != config_hash or
                    saved_meta.get("provenance") != fold_provenance or
                    sha256(checkpoint) != saved_meta.get("checkpoint_sha256") or
                    sha256(history_path) != saved_meta.get("history_sha256")):
                raise ValueError(f"Pilot checkpoint is stale or corrupt: {checkpoint}")
        elif resume and fold in completed_folds:
            raise ValueError(f"Completed fold checkpoint is missing: {checkpoint}")

        def checkpoint_receipt(_state):
            with np.load(checkpoint, allow_pickle=False) as saved:
                required = {"theta", "head", "bias", "m_t", "v_t", "m_h", "v_h", "m_b", "v_b", "rng_state", "provenance", "epoch"}
                if not required.issubset(saved.files):
                    raise ValueError("Atomic pilot checkpoint lacks model, Adam, RNG, epoch, or provenance state")
                epoch = int(saved["epoch"])
                if json.loads(str(saved["provenance"].item())) != fold_provenance:
                    raise ValueError("Checkpoint provenance does not match the current fold")
            atomic_json(sidecar, {"run_config_sha256": config_hash, "fold": fold, "epoch": epoch,
                "provenance": fold_provenance, "checkpoint_sha256": sha256(checkpoint),
                "history_sha256": sha256(history_path), "optimizer_state": "Adam moments persisted",
                "rng_state_persisted": True})

        os.environ["LIGHT_QNN_RESUME"] = "1" if resume and checkpoint.exists() else "0"
        q.STEPS = invocation_epochs
        q.check_ram()
        train_start = time.perf_counter()
        theta, head, bias, history, local_seconds = q.fit_model(
            xt, y[train], model_config, SEED + fold, checkpoint, history_path,
            provenance=fold_provenance, checkpoint_callback=checkpoint_receipt,
            sample_weights=sample_weights)
        training_seconds = time.perf_counter() - train_start
        checkpoint_receipt({})
        saved_history = json.loads(history_path.read_text(encoding="utf-8"))
        with np.load(checkpoint, allow_pickle=False) as saved:
            completed_epoch = int(saved["epoch"])
            optimizer_step_count = int(np.sum(np.abs(saved["m_t"])) + np.sum(np.abs(saved["v_t"])) > 0)
            if completed_epoch != invocation_epochs:
                raise ValueError(f"Fold {fold} stopped at epoch {completed_epoch}; expected {invocation_epochs}")
        fold_results.append({"fold": fold, "train_records": len(train), "validation_records": len(valid),
            "train_groups": len(set(groups[i] for i in train)), "validation_groups": len(set(groups[i] for i in valid)),
            "training_seconds_this_invocation": training_seconds, "training_seconds_fit_reported": local_seconds,
            "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": sha256(checkpoint),
            "completed_epoch": completed_epoch, "history_epochs": [item["epoch"] for item in saved_history],
            "optimizer_state_nonzero": bool(optimizer_step_count), "available_ram_mib": q.check_ram(),
            "gradient_check": q.gradient_check(model_config)})
        if completed_epoch == FINAL_EPOCHS:
            prob = q.predict_prob(xv, theta, head, bias, model_config)
            pred = (prob >= 0.5).astype(int)
            for j, index in enumerate(valid):
                qnn_oof.append({"record_id": ids[index], "group_id": groups[index], "fold": fold,
                                "y_true": int(y[index]), "probability": float(prob[j]), "y_pred": int(pred[j])})
        completed_folds[fold] = {"fold": fold, "epoch": completed_epoch,
            "checkpoint_sha256": sha256(checkpoint), "history_sha256": sha256(history_path)}
        atomic_json(run_manifest_path, {"schema": "source-isolated-qnn-pilot-v1",
            "status": "running", "config": config, "config_sha256": config_hash,
            "source_audit": sources, "completed_folds": [completed_folds[k] for k in sorted(completed_folds)],
            "fold_results": fold_results, "classical_baseline_sha256": sha256(baseline_path),
            "resume_events": int((prior or {}).get("resume_events", 0)) + int(resume),
            "frozen_test_accessed": False})

    available_end = round(psutil.virtual_memory().available / 2**20, 1)
    if invocation_epochs < FINAL_EPOCHS:
        report = {"schema": "source-isolated-qnn-pilot-v1", "status": "paused_checkpointed",
                  "config": config, "config_sha256": config_hash, "source_audit": sources,
                  "completed_folds": [completed_folds[k] for k in sorted(completed_folds)],
                  "fold_results": fold_results, "pause_epoch": invocation_epochs,
                  "resume_events": int((prior or {}).get("resume_events", 0)) + int(resume),
                  "elapsed_seconds_this_invocation": round(time.perf_counter() - invocation_start, 3),
                  "memory_available_start_mib": available_start, "memory_available_end_mib": available_end,
                  "frozen_test_accessed": False}
        atomic_json(run_manifest_path, report)
        return report

    if len(qnn_oof) != len(rows) or len({row["record_id"] for row in qnn_oof}) != len(rows):
        raise ValueError("Completed QNN OOF predictions do not cover each development record exactly once")
    qnn_oof.sort(key=lambda row: row["record_id"])
    baseline_oof.sort(key=lambda row: row["record_id"])
    qnn_digest = atomic_csv(OUT / "qnn_oof_predictions.csv", qnn_oof,
        ["record_id", "group_id", "fold", "y_true", "probability", "y_pred"])
    y_all = [row["y_true"] for row in qnn_oof]
    qnn_metrics = metric_dict(y_all, [row["y_pred"] for row in qnn_oof])
    baseline_metrics = metric_dict([int(row["y_true"]) for row in baseline_oof],
                                    [int(row["y_pred"]) for row in baseline_oof])
    fold_metrics = []
    for fold in folds:
        qp = [r for r in qnn_oof if int(r["fold"]) == fold]
        bp = [r for r in baseline_oof if int(r["fold"]) == fold]
        fold_metrics.append({"fold": fold, "qnn": metric_dict([r["y_true"] for r in qp], [r["y_pred"] for r in qp]),
                             "classical": metric_dict([int(r["y_true"]) for r in bp], [int(r["y_pred"]) for r in bp])})
    # Refit fold 1 once without interruption in a temporary directory and
    # compare every trainable and Adam/RNG value with the interrupted/resumed
    # checkpoint. This is a direct continuity check, not an evaluation run.
    fold = folds[0]
    valid = np.asarray([i for i, rid in enumerate(ids) if assignment[rid][0] == fold], dtype=int)
    train, valid = build_fold_indices(rows, assignment, fold, training_source)
    xt, _, _ = q.minmax_fit_apply(raw_x[train].tolist(), raw_x[valid].tolist())
    q.STEPS = FINAL_EPOCHS
    os.environ["LIGHT_QNN_RESUME"] = "0"
    with tempfile.TemporaryDirectory(prefix="qnn-pilot-uninterrupted-reference-") as temp_dir:
        reference_checkpoint = Path(temp_dir) / "checkpoint.npz"
        reference_weights = None
        if TASK_CONFIGS[TASK].get("class_weighting") == "balanced":
            reference_weights = training_fold_balanced_sample_weights(y[train])
        q.fit_model(np.asarray(xt), y[train], model_config, SEED + fold,
                    reference_checkpoint, Path(temp_dir) / "history.json",
                    provenance={"reference_run_config_sha256": config_hash, "fold": fold},
                    sample_weights=reference_weights)
        with np.load(reference_checkpoint, allow_pickle=False) as reference, \
             np.load(qnn_folder / f"fold_{fold}" / "checkpoint.npz", allow_pickle=False) as resumed:
            parity_fields = ["theta", "head", "bias", "m_t", "v_t", "m_h", "v_h", "m_b", "v_b", "rng_state"]
            parity = {name: bool(np.array_equal(reference[name], resumed[name])) for name in parity_fields}
    os.environ["LIGHT_QNN_RESUME"] = "1"
    if not all(parity.values()):
        raise ValueError(f"Interrupted/resumed training differs from uninterrupted reference: {parity}")
    bootstrap = paired_group_bootstrap(qnn_oof, baseline_oof, seed=SEED, iterations=1000)
    baseline_summary = {"model": "balanced LogisticRegression", **baseline_metrics,
                        "features": "same 8 hashed-text features; fold-local train-only minmax scaling",
                        "predictions_sha256": sha256(baseline_path)}
    source_by_id = {row["record_id"]: str(row.get("source", "dataset")) for row in rows}
    sources_present = sorted(set(source_by_id.values()))
    qnn_by_source = {source: metric_dict(
        [row["y_true"] for row in qnn_oof if source_by_id[row["record_id"]] == source],
        [row["y_pred"] for row in qnn_oof if source_by_id[row["record_id"]] == source])
        for source in sources_present}
    baseline_by_source = {source: metric_dict(
        [int(row["y_true"]) for row in baseline_oof if source_by_id[row["record_id"]] == source],
        [int(row["y_pred"]) for row in baseline_oof if source_by_id[row["record_id"]] == source])
        for source in sources_present}
    result = {"status": "completed", "task": TASK, "target_semantics": TASK_CONFIGS[TASK]["target_semantics"],
              "rows": len(rows), "groups": len(set(groups)), "folds": len(folds),
              "class_counts": dict(Counter(map(str, y))), "qnn": {"model": model_config, **qnn_metrics,
                  "predictions_sha256": qnn_digest, "fold_metrics": fold_metrics,
                  "checkpoints": [completed_folds[k] for k in sorted(completed_folds)]},
              "classical": baseline_summary, "qnn_metrics_by_source": qnn_by_source,
              "classical_metrics_by_source": baseline_by_source,
              "training_source_filter": training_source, "transfer_target_source": transfer_target_source,
              "paired_group_bootstrap_95_percentile": bootstrap,
              "resume_verification": {"status": "verified", "resume_events": int((prior or {}).get("resume_events", 0)) + int(resume),
                  "checkpoint_epochs": [int(completed_folds[k]["epoch"]) for k in sorted(completed_folds)],
                  "all_optimizer_states_present": all(bool(x["optimizer_state_nonzero"]) for x in fold_results),
                  "uninterrupted_reference_parity": parity,
                  "all_group_folds_preserved": True},
              "runtime": {"elapsed_seconds_this_invocation": round(time.perf_counter() - invocation_start, 3),
                  "memory_available_start_mib": available_start, "memory_available_end_mib": available_end,
                  "memory_floor_mib": RAM_FLOOR_MIB, "batch_size": BATCH_SIZE},
              "source_audit": sources, "config": config, "config_sha256": config_hash,
              "frozen_test_accessed": False, "evaluation_scope": "development grouped OOF only"}
    atomic_json(OUT / "metrics.json", result)
    final_manifest = {**result, "status": "completed", "completed_folds": [completed_folds[k] for k in sorted(completed_folds)],
                      "fold_results": fold_results, "classical_baseline_sha256": sha256(baseline_path),
                      "qnn_oof_sha256": qnn_digest, "resume_events": result["resume_verification"]["resume_events"]}
    atomic_json(run_manifest_path, final_manifest)
    return final_manifest


def paired_group_bootstrap(qnn_oof: list[dict], classical_oof: list[dict], *, seed: int, iterations: int) -> dict:
    rng = np.random.default_rng(seed)
    q_by_group: dict[str, list[dict]] = defaultdict(list)
    c_by_group: dict[str, list[dict]] = defaultdict(list)
    for row in qnn_oof: q_by_group[row["group_id"]].append(row)
    for row in classical_oof: c_by_group[row["group_id"]].append(row)
    if set(q_by_group) != set(c_by_group): raise ValueError("QNN/classical bootstrap groups differ")
    groups = sorted(q_by_group)
    differences = []
    for _ in range(iterations):
        sample = rng.choice(groups, size=len(groups), replace=True)
        q_rows = [row for group in sample for row in q_by_group[str(group)]]
        c_rows = [row for group in sample for row in c_by_group[str(group)]]
        qm = metric_dict([r["y_true"] for r in q_rows], [r["y_pred"] for r in q_rows])["macro_f1"]
        cm = metric_dict([int(r["y_true"]) for r in c_rows], [int(r["y_pred"]) for r in c_rows])["macro_f1"]
        differences.append(qm - cm)
    return {"groups_resampled": len(groups), "iterations": iterations,
            "macro_f1_difference_qnn_minus_classical": float(np.mean(differences)),
            "percentile_95_ci": [float(np.quantile(differences, .025)), float(np.quantile(differences, .975))]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASK_CONFIGS), default="bugs4q_na_source_state")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int, default=FINAL_EPOCHS)
    parser.add_argument("--stop-after-epochs", type=int)
    args = parser.parse_args()
    configure_task(args.task)
    if args.epochs != FINAL_EPOCHS:
        raise SystemExit(f"This registered pilot has a fixed epoch budget of {FINAL_EPOCHS}")
    if args.stop_after_epochs is not None and args.stop_after_epochs >= FINAL_EPOCHS:
        raise SystemExit("stop-after-epochs must be below the final epoch budget")
    result = run(args.stop_after_epochs, args.resume)
    print(json.dumps({"status": result["status"], "output": str(OUT.relative_to(ROOT)),
                      "completed_folds": len(result.get("completed_folds", [])),
                      "resume_events": result.get("resume_events", 0),
                      "frozen_test_accessed": result.get("frozen_test_accessed")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
