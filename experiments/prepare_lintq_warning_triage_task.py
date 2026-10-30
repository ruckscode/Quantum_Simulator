"""Prepare a grouped LintQ warning TP-vs-FP triage dataset from manual labels."""
from __future__ import annotations

import ast
import csv
import io
import json
import os
import re
import tokenize
from collections import Counter
from pathlib import Path

from prepare_qolumbina_detection_task import strip_comments_and_docstrings

ROOT = Path(__file__).resolve().parents[1]
LINTQ = ROOT / "data/external_quantum_datasets/lintq"
OUT = ROOT / "reports/experiments/qmutbench_multisource_qml_campaign_20261028/lintq_warning_triage"


def extended(path: Path) -> str:
    value = str(path.resolve())
    return "\\\\?\\" + value if os.name == "nt" and not value.startswith("\\\\?\\") else value


def main() -> None:
    ann_path = LINTQ / "bug_reports/Inspection_LintQ_on_LintQ_dataset.csv"
    source_root = extended(LINTQ / "extracted/exp_v08/files_selected")
    source_files = {}
    for base, dirs, files in os.walk(source_root):
        for name in files:
            source_files.setdefault(name.casefold(), os.path.join(base, name))
    with ann_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        review_title = next(reader)
        headers = next(reader)
        rows = list(csv.DictReader(f, fieldnames=headers))
    all_labels = Counter(x["label_resolution"].strip() for x in rows)
    selected = [x for x in rows if x["label_resolution"].strip() in {"TP", "FP"}]
    output = []
    missing_sources = []
    for i, item in enumerate(selected):
        name = item["file"].strip()
        src_path = source_files.get(name.casefold())
        if src_path is None:
            missing_sources.append(name)
            continue
        try:
            with open(src_path, encoding="utf-8", errors="replace") as f:
                code = strip_comments_and_docstrings(f.read())
        except OSError:
            missing_sources.append(name)
            continue
        lines = code.splitlines()
        try:
            line_no = int(float(item["line"]))
        except (TypeError, ValueError):
            line_no = 1
        lo, hi = max(1, line_no - 10), min(len(lines), line_no + 10)
        context = "\n".join(lines[lo-1:hi])
        message = re.sub(r"\s*\(l:\s*\d+,\s*c:\s*\d+\)", "", item["message"])
        group = item["issue_local"].strip() or f"file:{name}"
        output.append({"record_id": f"lintq-warning-{i+1}", "group_key": group,
            "label": int(item["label_resolution"].strip() == "TP"),
            "code": f"rule_{item['rule_id']}\nwarning_{message}\n{context}",
            "source_file": name, "source_line": line_no, "rule_id": item["rule_id"],
            "warning_message": message, "label_resolution": item["label_resolution"].strip()})
    groups = sorted({x["group_key"] for x in output})
    group_ids = {g: i + 1 for i, g in enumerate(groups)}
    for row in output:
        row["bug_id"] = group_ids[row["group_key"]]
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "normalized_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0]))
        writer.writeheader(); writer.writerows(output)
    class_counts = Counter(x["label"] for x in output)
    definition = {"status": "task_manifest_prepared",
        "task": "classify manually annotated LintQ warnings as true positive versus false positive",
        "label_map": {"TP": 1, "FP": 0}, "excluded_label_counts": {k: v for k, v in all_labels.items() if k not in {"TP", "FP"}},
        "source_artifact": str(LINTQ.relative_to(ROOT)), "annotation_file": str(ann_path.relative_to(ROOT)),
        "annotation_rows": len(rows), "selected_rows": len(output), "class_counts": {str(k): v for k, v in class_counts.items()},
        "groups": len(groups), "group_unit": "shared issue_local annotation group where supplied, otherwise source file; no source file can cross folds",
        "source_files": len({x["source_file"] for x in output}), "missing_source_files": sorted(set(missing_sources)),
        "feature_policy": "rule identifier, warning message, and a 21-line code context centered on the warning; code comments/docstrings stripped; annotation reason, reported_as, label_resolution, and issue_local excluded from features",
        "limitations": ["228 annotation groups and 290 selected warnings", "Manual TP/FP labels are not a full-corpus program label", "NW cases are ambiguous and excluded", "This task evaluates warning triage, not buggy-versus-fixed program classification"]}
    (OUT / "task_definition.json").write_text(json.dumps(definition, indent=2), encoding="utf-8")
    print(json.dumps({k: definition[k] for k in ("task", "annotation_rows", "selected_rows", "class_counts", "groups", "source_files", "missing_source_files", "excluded_label_counts")}, indent=2))


if __name__ == "__main__":
    main()
