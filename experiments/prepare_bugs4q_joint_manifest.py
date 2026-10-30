"""Combine verified non-overlapping Bugs4Q core and Bugs4Q-NA source pairs."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/experiments/qmutbench_multisource_qml_campaign_20261028/bugs4q_joint"
SOURCES = {
    "Bugs4Q_core": ROOT / "reports/experiments/qmutbench_multisource_qml_campaign_20261028/bugs4q_standalone/normalized_manifest.csv",
    "Bugs4Q_NA_additional": ROOT / "reports/experiments/qmutbench_multisource_qml_campaign_20261028/bugs4q_na/normalized_manifest.csv",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for source, path in SOURCES.items():
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                source_gid = str(row["bug_id"])
                gid = str(int(hashlib.sha256(f"{source}:{source_gid}".encode()).hexdigest()[:12], 16))
                rows.append({"record_id": f"{source}:{row['record_id']}", "bug_id": gid,
                             "label": int(row["label"]), "code": row["code"], "source": source,
                             "source_group_id": source_gid})
    with (OUT / "normalized_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    groups = {}
    for row in rows:
        groups.setdefault(row["source"], set()).add(row["bug_id"])
    (OUT / "preparation_manifest.json").write_text(json.dumps({
        "dataset": "Pooled Bugs4Q core + non-overlapping Bugs4Q-NA Python issue pairs",
        "target_semantics": "fixed=0, buggy=1, source-native meaning shared by both sources",
        "features": "Same source-code tokenization and feature pipeline; comments and docstrings excluded",
        "rows": len(rows), "groups": len({r["bug_id"] for r in rows}),
        "per_source_rows": {s: sum(r["source"] == s for r in rows) for s in SOURCES},
        "per_source_groups": {s: len(g) for s, g in groups.items()},
        "overlap_control": "Bugs4Q-NA pairs with exact README issue URL overlap against the original core subset were excluded during source-specific preparation; group IDs are namespaced by source.",
        "split": "5-fold GroupKFold across the pooled set; each buggy/fixed pair remains within its issue group.",
        "integration_question": "Does jointly fitting on both compatible sources improve pooled development-fold OOF performance versus the separate-source runs?",
    }, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "groups": len({r['bug_id'] for r in rows}),
                      "per_source_rows": {s: sum(r['source'] == s for r in rows) for s in SOURCES}}, indent=2))


if __name__ == "__main__":
    main()
