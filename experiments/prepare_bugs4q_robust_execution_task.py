"""Prepare a group-safe pass/fail task from Bugs4Q-Robust source snapshots."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.bugs4q_standalone_benchmark import sanitized_source

SOURCE = ROOT / "data/Bugs4Q_Robust_replication/bugs4q-robust_snapshot"
OUT = ROOT / "reports/experiments/qmutbench_multisource_qml_campaign_20261028/bugs4q_robust_execution"


def main() -> None:
    rows = []
    paired_runs = []
    with (SOURCE / "execution_results.csv").open(encoding="utf-8-sig", newline="") as f:
        results = list(csv.DictReader(f))
    for result in results:
        issue, run = result["id"], result["run"]
        folder = SOURCE / "bugs4q-robust" / issue
        test_path = folder / "test.py"
        if not test_path.is_file():
            raise FileNotFoundError(test_path)
        test = sanitized_source(test_path)
        pair = []
        for state, field in (("buggy", "test_buggy"), ("fixed", "test_fixed")):
            program_path = folder / f"{state}.py"
            if not program_path.is_file():
                raise FileNotFoundError(program_path)
            outcome = result[field]
            if outcome not in {"Pass", "Fail"}:
                continue
            label = int(outcome == "Fail")
            pair.append(label)
            rows.append({
                "record_id": f"issue-{issue}-run-{run}-{state}",
                "bug_id": int(issue),
                "label": label,
                "code": "TEST_CODE " + test + " PROGRAM_CODE " + sanitized_source(program_path),
                "source_path": str(program_path.relative_to(ROOT)),
                "test_path": str(test_path.relative_to(ROOT)),
                "source_state": state,
                "run_index_excluded_from_features": run,
            })
        if len(pair) == 2:
            paired_runs.append({"bug_id": issue, "run": run, "buggy_fail": pair[0], "fixed_fail": pair[1]})
    OUT.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "normalized_manifest.csv", index=False)
    summary = {
        "status": "task_manifest_prepared",
        "task": "predict whether the published Bugs4Q-Robust test passes or fails on a program snapshot",
        "label_map": {"Pass": 0, "Fail": 1},
        "source_artifact": "data/Bugs4Q_Robust_replication/bugs4q-robust_snapshot/bugs4q-robust/<issue>/{buggy,fixed,test}.py",
        "execution_records": len(results),
        "paired_execution_records": len(paired_runs),
        "observations": int(len(frame)),
        "issues": int(frame.bug_id.nunique()),
        "class_counts": {str(k): int(v) for k, v in frame.label.value_counts().sort_index().items()},
        "group_split_unit": "issue id; all buggy/fixed snapshots and repeated runs for an issue remain together",
        "feature_policy": "test harness source plus the tested program snapshot; comments/docstrings stripped; issue id, run index, and outcome log excluded",
        "outcome_pair_counts_by_label": {
            "buggy_fail_fixed_pass": sum(x["buggy_fail"] == 1 and x["fixed_fail"] == 0 for x in paired_runs),
            "buggy_pass_fixed_fail": sum(x["buggy_fail"] == 0 and x["fixed_fail"] == 1 for x in paired_runs),
            "both_fail": sum(x["buggy_fail"] == 1 and x["fixed_fail"] == 1 for x in paired_runs),
            "both_pass": sum(x["buggy_fail"] == 0 and x["fixed_fail"] == 0 for x in paired_runs),
        },
        "limitations": [
            "Task predicts the harness execution result, not a universal bug label.",
            "Repeated runs for a given issue are dependent and therefore grouped together in all folds.",
            "Only the 37 issues with execution rows in this artifact are included.",
        ],
    }
    (OUT / "task_definition.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
