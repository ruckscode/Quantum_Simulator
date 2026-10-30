"""Build a context-rich LintQ task with exact-feature components kept in one fold."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/experiments/qmutbench_multisource_qml_campaign_20261028/lintq_warning_triage/normalized_manifest.csv"
OUT = ROOT / "reports/experiments/qmutbench_multisource_qml_campaign_20261028/lintq_context_v2"
CONTRACT_PATH = ROOT / "experiments/source_task_contracts/lintq_warning_triage_context_v2.json"
SEED = 20261029


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_csv(path: Path, rows: list[dict], fields: list[str]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader(); writer.writerows(rows); stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)
    return sha256(path)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True); stream.write("\n")
            stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def main() -> None:
    with SOURCE.open(encoding="utf-8-sig", newline="") as stream:
        source_rows = list(csv.DictReader(stream))
    if len(source_rows) != 290 or len({r["record_id"] for r in source_rows}) != 290:
        raise ValueError("Expected 290 unique accepted LintQ warning records")
    for row in source_rows:
        label = int(row["label"])
        resolution = row["label_resolution"].strip().upper()
        if label not in (0, 1) or resolution != ("TP" if label else "FP"):
            raise ValueError(f"Manual label/resolution mismatch for {row['record_id']}")
        if not row["code"].strip() or not row["bug_id"].strip():
            raise ValueError(f"Missing model context or source group for {row['record_id']}")

    # Component union joins both the original issue group and any identical
    # model-input context. Thus duplicate contexts cannot cross a fold even
    # when they were attached to different source issue IDs.
    parent = list(range(len(source_rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    seen: dict[tuple[str, str], int] = {}
    for i, row in enumerate(source_rows):
        code_key = hashlib.sha256(" ".join(row["code"].split()).encode("utf-8")).hexdigest()
        for key in (("issue", row["bug_id"]), ("context", code_key)):
            if key in seen:
                union(i, seen[key])
            else:
                seen[key] = i
    members: dict[int, list[int]] = defaultdict(list)
    for i in range(len(source_rows)):
        members[find(i)].append(i)
    component_by_index = {}
    for indices in members.values():
        component = "lintq-component-" + hashlib.sha256(
            "\n".join(sorted(source_rows[i]["record_id"] for i in indices)).encode("utf-8")
        ).hexdigest()[:20]
        for i in indices:
            component_by_index[i] = component
    rows = [{**row, "model_group_id": component_by_index[i]} for i, row in enumerate(source_rows)]
    y = [int(row["label"]) for row in rows]
    groups = [row["model_group_id"] for row in rows]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    split_rows = []
    fold_counts = {}
    for fold, (_, valid) in enumerate(splitter.split(rows, y, groups), 1):
        fold_counts[str(fold)] = {"rows": len(valid), "groups": len({groups[i] for i in valid}),
                                  "class_counts": {str(c): sum(y[i] == c for i in valid) for c in (0, 1)}}
        split_rows.extend({"model": "lintq_context_v2", "record_id": rows[i]["record_id"],
                           "group_id": groups[i], "fold": fold, "y_true": y[i]} for i in valid)
    fold_by_id = {row["record_id"]: row["fold"] for row in split_rows}
    if len(fold_by_id) != len(rows) or any(
        len({fold_by_id[r["record_id"]] for r in rows if r["model_group_id"] == group}) != 1
        for group in set(groups)
    ):
        raise ValueError("Component group split is incomplete or leaks across folds")
    feature_groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        feature_hash = hashlib.sha256(" ".join(row["code"].split()).encode("utf-8")).hexdigest()
        feature_groups[feature_hash].append(row)
    label_conflict_components = [items for items in feature_groups.values()
                                 if len({int(r["label"]) for r in items}) > 1]
    if any(len({fold_by_id[r["record_id"]] for r in items}) != 1 for items in feature_groups.values()):
        raise ValueError("Identical context fingerprints leaked across new folds")

    OUT.mkdir(parents=True, exist_ok=True)
    input_path = OUT / "normalized_manifest.csv"
    fold_path = OUT / "group_folds.csv"
    input_sha = atomic_csv(input_path, rows, list(rows[0]))
    fold_sha = atomic_csv(fold_path, split_rows, ["model", "record_id", "group_id", "fold", "y_true"])
    contract = {"source_id": "lintq", "task_id": "warning_triage_context_v2",
        "target_semantics": "Manual LintQ true-positive (1) versus false-positive (0) warning triage using the rule, warning text, and source-code context. Not hardware diagnosis.",
        "required": ["record_id", "model_group_id", "label", "code", "rule_id", "warning_message", "label_resolution"],
        "record_id_column": "record_id", "group_column": "model_group_id", "label_column": "label",
        "feature_column": "code", "model_feature_fields": ["code"], "allowed_labels": ["0", "1"],
        "fold_record_column": "record_id", "fold_group_column": "group_id", "fold_column": "fold",
        "fold_model_column": "model", "fold_model_value": "lintq_context_v2", "fold_target_column": "y_true",
        "fold_label_map": {"0": 0, "1": 1}, "input_path": str(input_path.relative_to(ROOT)),
        "fold_path": str(fold_path.relative_to(ROOT)), "output_path": "data/processed/lintq/warning_triage_context_v2",
        "upstream_source_path": str(SOURCE.relative_to(ROOT)), "upstream_source_sha256": sha256(SOURCE),
        "grouping_policy": "Connected components of original warning issue group and exact normalized code-context fingerprint; no issue or exact model-input component crosses the five stratified folds.",
        "split_seed": SEED, "feature_policy": "code column contains rule identifier, warning message, and source snippet; source path, line, resolution, and bug identifier are excluded from model inputs."}
    atomic_json(CONTRACT_PATH, contract)
    report = {"status": "PREPARED_FOR_SOURCE_ISOLATED_VALIDATION", "task_id": contract["task_id"],
        "accepted_input_rows": len(rows), "rejected_rows": 0, "original_issue_groups": len({r["bug_id"] for r in rows}),
        "component_groups": len(set(groups)), "exact_context_fingerprints": len(feature_groups),
        "cross_fold_exact_context_fingerprints": 0, "same_context_conflicting_label_components": len(label_conflict_components),
        "same_context_conflicting_label_rows": sum(len(items) for items in label_conflict_components),
        "fold_counts": fold_counts, "upstream_source_sha256": contract["upstream_source_sha256"],
        "derived_input_sha256": input_sha, "fold_manifest_sha256": fold_sha,
        "label_policy": "All 290 upstream accepted labels retained; the three contradictory rows sharing identical full context remain grouped in one fold and are documented as irreducible feature-label ambiguity.",
        "frozen_test_accessed": False}
    atomic_json(OUT / "preparation_report.json", report)
    atomic_json(CONTRACT_PATH, contract)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
