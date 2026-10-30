"""Export the established development-only grouped folds for remote QML runs.

The bundle contains only the 578,117 saved development record IDs, binary labels,
origin names, and fold assignments. It never includes frozen-test records.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np

EXPECTED_ROWS = 578_117
EXPECTED_ORIGINS = 305
EXPECTED_SOURCE_SHA256 = "3f4547868673c583f047cee19d27c7bc97c64f15bb0586098ada778a3a0ff778"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_bundle(root: Path, output: Path) -> dict:
    root, output = root.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing reference bundle: {output}")
    archive = root / "data/qmutbench_ml/exhaustive_campaign_v1/five_fold_confirmation/mutation_diff_oof.npz"
    oof_csv = root / "data/qmutbench_ml/master_optimization_v1/oof_predictions.csv"
    source = root / "data/qmutbench_ingestion/raw/official_all_data_20261015.csv"
    if sha256(source) != EXPECTED_SOURCE_SHA256:
        raise ValueError("Pinned QMutBench metadata checksum mismatch")
    with np.load(archive, allow_pickle=False) as saved:
        required = {"record_id", "y", "fold"}
        if not required <= set(saved.files):
            raise ValueError(f"Saved fold archive lacks numeric arrays: {required - set(saved.files)}")
        record_ids = np.asarray(saved["record_id"], dtype=np.int64)
        labels = np.asarray(saved["y"], dtype=np.int8)
        folds = np.asarray(saved["fold"], dtype=np.int8)
    if not (record_ids.ndim == labels.ndim == folds.ndim == 1 and
            len(record_ids) == len(labels) == len(folds) == EXPECTED_ROWS):
        raise ValueError("Saved development fold archive does not have the expected 578,117 rows")
    if len(np.unique(record_ids)) != EXPECTED_ROWS or set(np.unique(labels)) != {0, 1} or set(np.unique(folds)) != {1, 2, 3, 4, 5}:
        raise ValueError("Saved development IDs, labels, or five-fold assignments are invalid")

    expected = {int(rid): (int(y), int(fold)) for rid, y, fold in zip(record_ids, labels, folds)}
    seen: set[int] = set()
    origin_folds: dict[str, int] = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="qmutbench-qml-reference-") as temp_name:
        temp = Path(temp_name)
        manifest_path = temp / "reference_dev_oof.csv"
        with oof_csv.open("r", encoding="utf-8-sig", newline="") as src, manifest_path.open(
            "w", encoding="utf-8", newline=""
        ) as dst:
            reader = csv.DictReader(src)
            required_cols = {"record_id", "origin_program", "y_true"}
            if not reader.fieldnames or not required_cols <= set(reader.fieldnames):
                raise ValueError("Saved development OOF file lacks required lineage columns")
            writer = csv.DictWriter(dst, fieldnames=["record_id", "origin_program", "y_true", "fold"])
            writer.writeheader()
            rows = 0
            for row in reader:
                rid = int(row["record_id"])
                if rid not in expected or rid in seen:
                    raise ValueError(f"Unexpected or duplicate development record ID: {rid}")
                label, fold = expected[rid]
                origin = str(row["origin_program"])
                if not origin or int(row["y_true"]) != label:
                    raise ValueError(f"Development label or origin mismatch at record {rid}")
                old_fold = origin_folds.setdefault(origin, fold)
                if old_fold != fold:
                    raise ValueError(f"Origin crosses saved folds: {origin}")
                writer.writerow({"record_id": rid, "origin_program": origin,
                                 "y_true": label, "fold": fold})
                seen.add(rid)
                rows += 1
        if rows != EXPECTED_ROWS or len(origin_folds) != EXPECTED_ORIGINS or seen != set(expected):
            raise ValueError("OOF manifest and saved fold archive do not align exactly")

        numeric_path = temp / "reference_dev_folds.npz"
        np.savez_compressed(numeric_path, record_id=record_ids, y=labels, fold=folds)
        metadata = {
            "schema": "qmutbench-development-reference-bundle-v1",
            "source_metadata_sha256": EXPECTED_SOURCE_SHA256,
            "source_archive_sha256": sha256(archive),
            "development_records": rows,
            "development_origins": len(origin_folds),
            "folds": 5,
            "target_map": {"Survivor": 0, "WOO": 1, "OPO": 1},
            "frozen_test_rows_included": 0,
            "files": {
                "reference_dev_oof.csv": sha256(manifest_path),
                "reference_dev_folds.npz": sha256(numeric_path),
            },
        }
        (temp / "bundle_manifest.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
            for name in ("bundle_manifest.json", "reference_dev_oof.csv", "reference_dev_folds.npz"):
                bundle.write(temp / name, arcname=name)
    result = {**metadata, "bundle_path": str(output), "bundle_sha256": sha256(output),
              "bundle_bytes": output.stat().st_size}
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(export_bundle(args.root, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
