"""Write a development-only origin manifest without inspecting test labels."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(root: Path, output: Path) -> dict:
    root, output = root.resolve(), output.resolve()
    oof_path = root / "data/qmutbench_ml/master_optimization_v1/oof_predictions.csv"
    raw_path = root / "data/qmutbench_ingestion/raw/official_all_data_20261015.csv"
    prior_sample = root / "reports/experiments/qmutbench_three_class_qnn_campaign_20261027/sample_manifest.csv"
    dev_ids: set[int] = set()
    with oof_path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            dev_ids.add(int(row["record_id"]))
    selected = set()
    with prior_sample.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("role", "train") == "train":
                selected.add(row["origin_program"])
    # The historic sample manifest may be row-level and omit a role field.
    if not selected:
        summary = json.loads((root / "reports/experiments/qmutbench_three_class_qnn_campaign_20261027/campaign_summary.json").read_text(encoding="utf-8"))
        selected = set(summary["sample"]["selected_origins"])

    counts: dict[str, Counter[str]] = defaultdict(Counter)
    with raw_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        # Pandas exposes this blank first-column header as ``Unnamed: 0``.
        id_i = header.index("") if header and header[0] == "" else header.index("Unnamed: 0")
        origin_i, label_i = header.index("Origin_program"), header.index("Killed")
        for row in reader:
            record_id = int(row[id_i])
            if record_id not in dev_ids:
                continue
            origin, label = row[origin_i], row[label_i]
            if label not in {"Survivor", "WOO", "OPO"}:
                raise ValueError(f"Unexpected development label {label!r}")
            counts[origin][label] += 1
    if sum(sum(origin_counts.values()) for origin_counts in counts.values()) != len(dev_ids):
        raise AssertionError("Development source rows do not align exactly with saved OOF IDs")
    rare = sorted(origin for origin, labels in counts.items() if labels["OPO"] > 0)
    reserved = sorted(set(rare) - selected)
    doc = {
        "schema": "qmutbench-three-class-confirmation-origin-manifest-v1",
        "source_sha256": sha256(raw_path),
        "development_oof_manifest_sha256": sha256(oof_path),
        "prior_campaign_sample_manifest_sha256": sha256(prior_sample),
        "target_labels_read_only_for_saved_development_ids": True,
        "frozen_test_target_labels_accessed": False,
        "target_map": {"Survivor": 0, "WOO": 1, "OPO": 2},
        "development_rows": len(dev_ids),
        "development_label_counts": dict(Counter({k: sum(c[k] for c in counts.values()) for k in ("Survivor", "WOO", "OPO")})),
        "development_origins_with_OPO": len(rare),
        "selected_exploration_origins": sorted(selected),
        "reserved_confirmation_origins": reserved,
        "reserved_origin_count": len(reserved),
        "selected_reserved_overlap": sorted(selected & set(reserved)),
        "per_reserved_origin_class_counts": {origin: dict(counts[origin]) for origin in reserved},
    }
    if len(rare) != 48 or len(selected) != 12 or len(reserved) != 36 or doc["selected_reserved_overlap"]:
        raise AssertionError("Origin counts or separation differ from the predeclared campaign")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return doc


if __name__ == "__main__":
    project = Path.cwd()
    destination = project / "reports/experiments/qmutbench_three_class_qnn_campaign_20261027/confirmation_origin_manifest.json"
    result = write_manifest(project, destination)
    print(json.dumps({"reserved_origin_count": result["reserved_origin_count"],
                      "reserved_confirmation_origins": result["reserved_confirmation_origins"],
                      "development_label_counts": result["development_label_counts"]}, indent=2))
