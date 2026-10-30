"""Create a duplicate-comment-component split for the platform annotation task."""
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
SOURCE = ROOT / "reports/experiments/qmutbench_multisource_qml_campaign_20261028/quantum_platform_223_comment_only/normalized_manifest.csv"
OUT = ROOT / "reports/experiments/qmutbench_multisource_qml_campaign_20261028/quantum_platform_comment_v2"
CONTRACT_PATH = ROOT / "experiments/source_task_contracts/quantum_platform_comment_v2.json"
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
    if len(source_rows) != 319 or len({r["record_id"] for r in source_rows}) != len(source_rows):
        raise ValueError("Expected 319 unique quantum-platform comment records")
    for row in source_rows:
        if row["label"] not in {"0", "1"} or not row["code"].strip() or not row["bug_id"].strip():
            raise ValueError(f"Invalid annotation/context/group at record {row['record_id']}")

    parent = list(range(len(source_rows)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        a, b = find(a), find(b)
        if a != b: parent[max(a, b)] = min(a, b)

    seen: dict[tuple[str, str], int] = {}
    comment_label_sets: dict[str, set[str]] = defaultdict(set)
    comment_rows: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(source_rows):
        comment_hash = hashlib.sha256(" ".join(row["code"].split()).encode("utf-8")).hexdigest()
        comment_label_sets[comment_hash].add(row["label"])
        comment_rows[comment_hash].append(i)
        for key in (("issue", row["bug_id"]), ("comment", comment_hash)):
            if key in seen: union(i, seen[key])
            else: seen[key] = i
    conflict_hashes = [key for key, labels in comment_label_sets.items() if len(labels) > 1]
    if conflict_hashes:
        raise ValueError(f"Repeated exact comment has conflicting labels: {len(conflict_hashes)} fingerprints")

    components: dict[int, list[int]] = defaultdict(list)
    for i in range(len(source_rows)): components[find(i)].append(i)
    group_by_index = {}
    for indexes in components.values():
        group_id = "platform-comment-component-" + hashlib.sha256(
            "\n".join(sorted(source_rows[i]["record_id"] for i in indexes)).encode("utf-8")
        ).hexdigest()[:20]
        for i in indexes: group_by_index[i] = group_id
    rows = [{**row, "comment_component_id": group_by_index[i]} for i, row in enumerate(source_rows)]
    y = [int(row["label"]) for row in rows]
    groups = [row["comment_component_id"] for row in rows]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    split_rows = []
    fold_counts = {}
    for fold, (_, valid) in enumerate(splitter.split(rows, y, groups), 1):
        fold_counts[str(fold)] = {"rows": len(valid), "groups": len({groups[i] for i in valid}),
            "class_counts": {str(cls): sum(y[i] == cls for i in valid) for cls in (0, 1)}}
        split_rows.extend({"model": "quantum_platform_comment_v2", "record_id": rows[i]["record_id"],
                           "group_id": groups[i], "fold": fold, "y_true": y[i]} for i in valid)
    fold_by_id = {row["record_id"]: row["fold"] for row in split_rows}
    if len(fold_by_id) != len(rows): raise ValueError("Split omitted source records")
    for rows_with_comment in comment_rows.values():
        if len({fold_by_id[source_rows[i]["record_id"]] for i in rows_with_comment}) != 1:
            raise ValueError("Repeated exact comments cross validation folds")

    OUT.mkdir(parents=True, exist_ok=True)
    derived_path, fold_path = OUT / "normalized_manifest.csv", OUT / "group_folds.csv"
    input_sha = atomic_csv(derived_path, rows, list(rows[0]))
    fold_sha = atomic_csv(fold_path, split_rows, ["model", "record_id", "group_id", "fold", "y_true"])
    contract = {"source_id": "quantum_platform_bugs", "task_id": "comment_only_classification_v2",
        "target_semantics": "Curated issue-comment annotation classification; labels do not denote hardware faults.",
        "required": ["record_id", "comment_component_id", "label", "code", "repo", "commit_hash"],
        "record_id_column": "record_id", "group_column": "comment_component_id", "label_column": "label",
        "feature_column": "code", "model_feature_fields": ["code"], "allowed_labels": ["0", "1"],
        "fold_record_column": "record_id", "fold_group_column": "group_id", "fold_column": "fold",
        "fold_model_column": "model", "fold_model_value": "quantum_platform_comment_v2",
        "fold_target_column": "y_true", "fold_label_map": {"0": 0, "1": 1},
        "input_path": str(derived_path.relative_to(ROOT)), "fold_path": str(fold_path.relative_to(ROOT)),
        "output_path": "data/processed/quantum_platform_bugs/comment_only_classification_v2",
        "upstream_source_path": str(SOURCE.relative_to(ROOT)), "upstream_source_sha256": sha256(SOURCE),
        "grouping_policy": "Connected components of the original issue/commit group and exact normalized comment text; no issue group or repeated exact comment crosses folds.",
        "split_seed": SEED, "feature_policy": "Comment text only; repo, commit hash, issue identifier, and target are excluded from model features."}
    atomic_json(CONTRACT_PATH, contract)
    report = {"status": "PREPARED_FOR_SOURCE_ISOLATED_VALIDATION", "task_id": contract["task_id"],
        "accepted_input_rows": len(rows), "rejected_rows": 0,
        "original_issue_groups": len({r["bug_id"] for r in rows}), "comment_component_groups": len(set(groups)),
        "exact_comment_fingerprints": len(comment_rows), "cross_fold_exact_comment_fingerprints": 0,
        "conflicting_label_comment_fingerprints": 0,
        "duplicate_comment_rows": sum(len(v) for v in comment_rows.values() if len(v) > 1),
        "fold_counts": fold_counts, "upstream_source_sha256": contract["upstream_source_sha256"],
        "derived_input_sha256": input_sha, "fold_manifest_sha256": fold_sha,
        "label_policy": "All upstream accepted annotations retained; exact duplicate comment labels were checked and agree.",
        "frozen_test_accessed": False}
    atomic_json(OUT / "preparation_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
