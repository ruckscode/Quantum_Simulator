"""Build a provenance-bound nine-source input bundle for grouped classical CV.

Inputs are reconstructed from already validated source-isolated partitions.
Each adapter contains only record id, group id, target, and model text, with
group ids normalized to the classical runner's ``bug_id`` field.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/experiments/qml_source_preprocessing_20261029/classical_nine_source_v2_inputs"
CONFIG = OUT / "classical_nine_source_v2_config.json"
SUITE_OUT = "reports/experiments/local_grouped_classical_suite_nine_source_v2_20261029"
TASK_CONTRACTS = [
    ("bugs4q_core", "bugs4q_core_source_state.json"),
    ("bugs4q_na", "bugs4q_na_source_state.json"),
    ("bugs4q_pooled", "bugs4q_joint_source_state.json"),
    ("qbugs_qasm_v2", "qbugs_correct_buggy_circuit_state_qasm_v2.json"),
    ("quantum_platform_comment_v2", "quantum_platform_comment_v2.json"),
    ("qolumbina_ideal", "qolumbina_ideal_detection.json"),
    ("lintq_context_v2_exploratory", "lintq_warning_triage_context_v2.json"),
    ("bugs4q_robust_execution", "bugs4q_robust_execution.json"),
    ("qsimbench_exploratory", "qsimbench_backend_fingerprint.json"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_csv(path: Path, rows: list[dict], fields: list[str]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader(); writer.writerows(rows); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
    return sha256(path)


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing adapter bundle: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    task_config, task_report = [], []
    for name, contract_name in TASK_CONTRACTS:
        contract_path = ROOT / "experiments/source_task_contracts" / contract_name
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        processed_dir = ROOT / contract["output_path"]
        manifest_path = processed_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("validation_pass"):
            raise ValueError(f"Source task is not validated: {contract_name}")
        input_path = ROOT / contract["input_path"]
        if sha256(input_path) != manifest["input_sha256"]:
            raise ValueError(f"Input source hash changed for {contract_name}")
        source_rows = []
        for part in manifest["partitions"]:
            path = ROOT / part["path"]
            if sha256(path) != part["sha256"]:
                raise ValueError(f"Partition hash mismatch: {path}")
            with path.open(encoding="utf-8-sig", newline="") as stream:
                partition = list(csv.DictReader(stream))
            if len(partition) != int(part["rows"]):
                raise ValueError(f"Partition row count mismatch: {path}")
            source_rows.extend(partition)
        ids, groups = set(), set()
        normalized = []
        for row in source_rows:
            rid = str(row.get(contract["record_id_column"], ""))
            group = str(row.get(contract["group_column"], ""))
            label = str(row.get(contract["label_column"], ""))
            code = str(row.get(contract["feature_column"], ""))
            if not rid or not group or label not in {"0", "1"} or not code.strip():
                raise ValueError(f"Invalid row in {contract_name}: {rid!r}")
            if rid in ids:
                raise ValueError(f"Duplicate record id in {contract_name}: {rid}")
            ids.add(rid); groups.add(group)
            normalized.append({"record_id": rid, "bug_id": group, "label": int(label), "code": code})
        if len(normalized) != int(manifest["counts"]["accepted"]):
            raise ValueError(f"Accepted count mismatch for {contract_name}")
        adapter_path = OUT / f"{name}.csv"
        adapter_sha = atomic_csv(adapter_path, normalized, ["record_id", "bug_id", "label", "code"])
        class_counts = dict(sorted(Counter(str(row["label"]) for row in normalized).items()))
        task_config.append({"name": name, "input": adapter_path.relative_to(ROOT).as_posix(),
            "sha256": adapter_sha, "rows": len(normalized), "groups": len(groups),
            "source_id": contract["source_id"], "task_id": contract["task_id"],
            "source_contract": contract_path.relative_to(ROOT).as_posix(),
            "source_contract_sha256": sha256(contract_path),
            "source_input_sha256": manifest["input_sha256"],
            "source_processed_manifest_sha256": sha256(manifest_path),
            "source_split_sha256": manifest["split_manifest"]["sha256"],
            "target_semantics": contract["target_semantics"],
            "class_counts": class_counts})
        task_report.append({"name": name, "rows": len(normalized), "groups": len(groups),
            "classes": class_counts, "adapter_sha256": adapter_sha,
            "source_id": contract["source_id"], "task_id": contract["task_id"],
            "semantic_caveat": ("Exploratory: three records in one identical-context component retain conflicting manual labels."
                if name == "lintq_context_v2_exploratory" else
                "Exploratory: quantized descriptors collide across held-out circuit settings with both backend labels."
                if name == "qsimbench_exploratory" else None)})
    config = {"schema": "nine-source-classical-suite-v2", "output": SUITE_OUT,
        "folds": 5, "split": "GroupKFold by validated source group; independent of the QNN-matched OOF split",
        "models": "20 estimators per task, including training-fold-only over/undersampling and continuous-space SMOTE",
        "tasks": task_config}
    CONFIG.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {"schema": "nine-source-classical-input-audit-v2", "tasks": task_report,
        "task_count": len(task_report), "row_count": sum(x["rows"] for x in task_report),
        "source_hashes_rechecked": True, "training_run_started": False,
        "frozen_test_accessed": False, "limitations": [
            "This classical suite uses its own GroupKFold split and is not a matched QNN comparison.",
            "LintQ context and QSimBench remain exploratory with the listed source-specific caveats."]}
    (OUT / "input_audit.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"config": CONFIG.relative_to(ROOT).as_posix(), "task_count": len(task_report),
        "total_rows": summary["row_count"], "tasks": task_report}, indent=2))


if __name__ == "__main__":
    main()
