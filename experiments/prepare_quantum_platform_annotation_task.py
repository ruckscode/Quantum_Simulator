"""Prepare a metadata-only real-bug versus false-positive task from annotation data."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/external_quantum_datasets/quantum_platform_bugs_223/extracted/artifacts/annotation_bugs.csv"
OUT = ROOT / "reports/experiments/qmutbench_multisource_qml_campaign_20261028/quantum_platform_223_comment_only"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(SOURCE).fillna("")
    raw = raw[raw.real.isin(["bug", "fp"]) & raw.comment.fillna("").astype(str).str.strip().ne("")].copy()
    raw["label"] = (raw.real == "bug").astype(int)  # false positive=0, manually retained bug=1
    raw["bug_id"] = raw.commit_hash.map(lambda h: str(int(hashlib.sha256(str(h).encode()).hexdigest()[:12], 16)))
    raw["record_id"] = raw.id.astype(str)
    raw["code"] = raw.comment.astype(str).map(lambda s: re.sub(r"\b(?:false\s+positive|bug(?:s|gy)?|defect(?:s|ive)?|fix(?:es|ed|ing)?)\b", "labelterm", s, flags=re.I))
    frame = raw[["record_id", "bug_id", "label", "code", "repo", "commit_hash"]]
    frame.to_csv(OUT / "normalized_manifest.csv", index=False)
    (OUT / "preparation_manifest.json").write_text(json.dumps({
        "dataset": "Quantum Computing Platforms bug annotations (Paltenghi & Pradel)",
        "task": "classify manually annotated real bug versus false positive from nonempty commit/annotator comments, with explicit bug/fix terms masked",
        "label_map": {"fp": 0, "bug": 1}, "rows": len(frame),
        "real_bug_rows": int((frame.label == 1).sum()),
        "false_positive_rows": int((frame.label == 0).sum()),
        "commit_groups": int(frame.bug_id.nunique()),
        "group_split": "grouped by fix commit hash",
        "feature_policy": "Only nonempty comment text is used; explicit bug, defect, fix, or false-positive terms are replaced with a neutral token. Repository and commit fields identify groups but are excluded from model text.",
        "limitations": "This predicts curator real/false-positive labels from comment text; it is not source-code classification. Rows without comments are excluded. Remaining semantic clues may still be present in the prose.",
        "source_csv": str(SOURCE.relative_to(ROOT)),
    }, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(frame), "groups": frame.bug_id.nunique(),
                      "counts": frame.label.value_counts().sort_index().to_dict()}, indent=2))


if __name__ == "__main__":
    main()
