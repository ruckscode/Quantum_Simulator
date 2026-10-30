"""Export compact, sampled development-only confirmation data for Kaggle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import psutil


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export(root: Path, output: Path) -> dict:
    root, output = root.resolve(), output.resolve()
    available = psutil.virtual_memory().available / 2**20
    if available < 900:
        raise MemoryError(f"Export requires >= 900 MiB pre-import headroom; found {available:.0f} MiB")
    from qmutbench_qml.data.loader import FEATURES, load_development_records
    from qmutbench_qml.training.run_three_class_campaign import LABEL_MAP, RAM_STOP_MIB, _sample_manifest
    if psutil.virtual_memory().available / 2**20 < RAM_STOP_MIB + 200:
        raise MemoryError("Insufficient post-import RAM headroom before scanning development data")
    train_origins, train_ids, train_manifest, audit = _sample_manifest(root)
    reserved = audit["reserved_confirmation_sample"]
    reserved_ids = [int(row["record_id"]) for row in reserved]
    if len(audit["reserved_confirmation_origins"]) != 36 or set(train_origins) & set(audit["reserved_confirmation_origins"]):
        raise AssertionError("Confirmation origins differ from the predeclared 12/36 split")
    if psutil.virtual_memory().available / 2**20 < RAM_STOP_MIB + 200:
        raise MemoryError("Insufficient post-import RAM headroom before feature export")
    frame, adapter_audit = load_development_records(root,
        origins=list(train_origins) + list(audit["reserved_confirmation_origins"]),
        record_ids=list(train_ids) + reserved_ids)
    frame["y3"] = frame.Killed.astype(str).map(LABEL_MAP).astype(int)
    train_set = set(map(int, train_ids))
    frame["role"] = frame.record_id.map(lambda value: "train" if int(value) in train_set else "validation")
    frame["y"] = frame.y3
    output.mkdir(parents=True, exist_ok=False)
    fields = ["record_id", "origin_program", "role", "y", *FEATURES]
    for role in ("train", "validation"):
        frame.loc[frame.role == role, fields].to_csv(output / f"{role}.csv", index=False)
    data_manifest = {
        "schema": "qmutbench-three-class-kaggle-confirmation-data-v1",
        "development_only": True, "frozen_test_rows": 0,
        "label_map": {"Survivor": 0, "WOO": 1, "OPO": 2},
        "sampling_seed": 20261027, "per_class_per_origin_cap": 6,
        "training_origins": sorted(train_origins),
        "validation_origins": audit["reserved_confirmation_origins"],
        "training_rows": int((frame.role == "train").sum()),
        "validation_rows": int((frame.role == "validation").sum()),
        "training_class_counts": {str(k): int(v) for k, v in frame.loc[frame.role == "train", "y"].value_counts().sort_index().items()},
        "validation_class_counts": {str(k): int(v) for k, v in frame.loc[frame.role == "validation", "y"].value_counts().sort_index().items()},
        "features": FEATURES,
        "source_sha256": adapter_audit["source_sha256"],
        "train_sha256": sha256(output / "train.csv"),
        "validation_sha256": sha256(output / "validation.csv"),
        "feature_selection_fit_scope": "training folds only by runner; no precomputed feature selection",
        "known_license": "GPL-3.0; check source license and Kaggle sharing settings",
    }
    (output / "data_manifest.json").write_text(json.dumps(data_manifest, indent=2), encoding="utf-8")
    return data_manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.root, args.output), indent=2))
