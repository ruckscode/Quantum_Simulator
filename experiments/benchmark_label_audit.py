"""Audit benchmark metadata without inventing diagnosis-category labels.

The audit is intentionally descriptive: QMutBench mutation/survival outcomes and
Bugs4Q bug metadata are not automatically hardware-vs-program diagnosis labels.
CSV files are read in chunks so the full QMutBench export can be inspected with
bounded memory.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


def norm(value: Any) -> str:
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


def audit_csv(path: Path, chunksize: int = 50000) -> dict:
    header = pd.read_csv(path, nrows=0)
    columns = [str(c) for c in header.columns]
    counts: dict[str, Counter] = {c: Counter() for c in columns}
    missing: Counter = Counter()
    rows = 0
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        rows += len(chunk)
        for col in columns:
            series = chunk[col]
            missing[col] += int(series.isna().sum())
            # Count labels only for small categorical columns; cap to prevent
            # accidentally producing a giant report from high-cardinality data.
            if series.nunique(dropna=True) <= 100:
                counts[col].update(series.dropna().astype(str).value_counts().to_dict())
    normalized = {norm(c): c for c in columns}
    target_candidates = [
        name for name in ("killed", "survival", "survivalrate", "mutationoperator",
                          "operator", "mutationtype", "status", "label", "groundtruth",
                          "diagnosiscategory", "faultclass")
        if name in normalized
    ]
    target_summary = {}
    for candidate in target_candidates:
        col = normalized[candidate]
        target_summary[col] = {
            "missing": int(missing[col]),
            "non_missing": int(rows - missing[col]),
            "value_counts": dict(counts[col]) if counts[col] else {},
            "interpretation": (
                "Candidate mutation outcome/metadata only; do not map to hardware-vs-program diagnosis without a documented independent label definition."
                if candidate in {"killed", "survival", "survivalrate", "mutationoperator", "operator", "mutationtype"}
                else "Inspect definition and provenance before using as a supervised target."
            ),
        }
    return {
        "path": str(path),
        "rows": rows,
        "columns": columns,
        "missing_by_column": {c: int(missing[c]) for c in columns},
        "candidate_label_columns": target_summary,
        "diagnosis_ground_truth_column_present": any(
            norm(c) in {"diagnosiscategory", "faultclass", "hardwarevsprogram", "diagnosislabel"}
            for c in columns
        ),
        "warning": "This script audits schema and metadata only. It does not infer or manufacture diagnosis labels.",
    }


def audit_bugs4q(root: Path) -> dict:
    if not root.exists():
        return {"path": str(root), "exists": False, "source_files": 0, "buggy_files": 0, "fixed_files": 0, "modify_files": 0}
    files = [p for p in root.rglob("*") if p.is_file()]
    names = [p.name.lower() for p in files]
    return {
        "path": str(root),
        "exists": True,
        "source_files": len(files),
        "buggy_files": sum(name.startswith("buggy_") for name in names),
        "fixed_files": sum(name.startswith("fixed_") for name in names),
        "modify_files": sum(name.startswith("modify_") for name in names),
        "diagnosis_ground_truth": "Not inferred from buggy/fixed source pairs or bug-type metadata. Any mapping requires a documented, case-level justification.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qmutbench-csv", help="Path to official QMutBench all_data.csv or another official CSV export")
    parser.add_argument("--bugs4q-root", help="Path to extracted official Bugs4Q framework/database")
    parser.add_argument("--output", default="reports/benchmark_label_audit.json")
    parser.add_argument("--chunksize", type=int, default=50000)
    args = parser.parse_args()
    if not args.qmutbench_csv and not args.bugs4q_root:
        parser.error("provide --qmutbench-csv and/or --bugs4q-root")
    report = {
        "report": "Benchmark label/schema audit",
        "qmutbench": audit_csv(Path(args.qmutbench_csv), args.chunksize) if args.qmutbench_csv else None,
        "bugs4q": audit_bugs4q(Path(args.bugs4q_root)) if args.bugs4q_root else None,
        "conclusion": (
            "Only use labels for the task they actually define. Mutation survival/killed outcomes can support a mutation-testing task, "
            "but do not establish whether a diagnosis engine correctly attributed a fault to program or hardware."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("\nSaved:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
