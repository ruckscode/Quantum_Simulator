"""Verify and install pinned public data plus the dev-only reference bundle."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np

EXPECTED_SOURCE_SHA256 = "3f4547868673c583f047cee19d27c7bc97c64f15bb0586098ada778a3a0ff778"
EXPECTED_SOURCE_COMMIT = "01f5e7700c051fe715fc368b307deaa7e6869ba8"
EXPECTED_ROWS = 578_117
EXPECTED_ORIGINS = 305


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def install_inputs(source_csv: Path, official_checkout: Path, bundle_zip: Path, root: Path) -> dict:
    source_csv, official_checkout, bundle_zip, root = map(Path.resolve,
        (source_csv, official_checkout, bundle_zip, root))
    import subprocess
    commit = subprocess.run(["git", "-C", str(official_checkout), "rev-parse", "HEAD"],
                            check=True, capture_output=True, text=True).stdout.strip()
    if commit != EXPECTED_SOURCE_COMMIT:
        raise ValueError(f"QMutBench source commit mismatch: {commit}")
    dirty = subprocess.run(["git", "-C", str(official_checkout), "status", "--porcelain"],
                           check=True, capture_output=True, text=True).stdout.strip()
    if dirty:
        raise ValueError("Pinned QMutBench checkout is modified")

    target_source = root / "data/qmutbench_ingestion/raw/official_all_data_20261015.csv"
    target_source.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target_source.parent, prefix="official_all_data_",
                                     suffix=".tmp", delete=False) as temporary:
        temporary_path = Path(temporary.name)
        pending = b""
        with source_csv.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                chunk = pending + chunk
                pending = b""
                if chunk.endswith(b"\r"):
                    pending, chunk = b"\r", chunk[:-1]
                temporary.write(chunk.replace(b"\r\n", b"\n"))
            if pending:
                temporary.write(pending)
        temporary.flush()
        os.fsync(temporary.fileno())
    os.replace(temporary_path, target_source)
    source_sha = sha256(target_source)
    if source_sha != EXPECTED_SOURCE_SHA256:
        raise ValueError(f"Normalized QMutBench metadata checksum mismatch: {source_sha}")

    dev_root = root / "data/qmutbench_ml/master_optimization_v1"
    fold_root = root / "data/qmutbench_ml/exhaustive_campaign_v1/five_fold_confirmation"
    dev_root.mkdir(parents=True, exist_ok=True)
    fold_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle_zip) as bundle:
        names = set(bundle.namelist())
        required = {"bundle_manifest.json", "reference_dev_oof.csv", "reference_dev_folds.npz"}
        if not required <= names:
            raise ValueError(f"Reference bundle missing members: {required - names}")
        metadata = json.loads(bundle.read("bundle_manifest.json"))
        if (metadata.get("schema") != "qmutbench-development-reference-bundle-v1" or
                metadata.get("source_metadata_sha256") != EXPECTED_SOURCE_SHA256 or
                metadata.get("development_records") != EXPECTED_ROWS or
                metadata.get("development_origins") != EXPECTED_ORIGINS or
                metadata.get("frozen_test_rows_included") != 0):
            raise ValueError("Reference bundle metadata does not match the development-only contract")
        for member in ("reference_dev_oof.csv", "reference_dev_folds.npz"):
            content = bundle.read(member)
            digest = hashlib.sha256(content).hexdigest()
            if metadata.get("files", {}).get(member) != digest:
                raise ValueError(f"Reference bundle member checksum mismatch: {member}")
            output = (dev_root / "oof_predictions.csv" if member.endswith(".csv")
                      else fold_root / "mutation_diff_oof.npz")
            output.write_bytes(content)

    # Independently check CSV coverage, fold support, target mapping and origin isolation.
    archive = fold_root / "mutation_diff_oof.npz"
    with np.load(archive, allow_pickle=False) as saved:
        ids, y, folds = (np.asarray(saved[k]) for k in ("record_id", "y", "fold"))
    if not (ids.shape == y.shape == folds.shape == (EXPECTED_ROWS,)) or len(np.unique(ids)) != EXPECTED_ROWS:
        raise ValueError("Installed reference numeric arrays have invalid coverage")
    y_by_id = np.full(int(ids.max()) + 1, -1, dtype=np.int8)
    fold_by_id = np.zeros(int(ids.max()) + 1, dtype=np.int8)
    y_by_id[ids] = y
    fold_by_id[ids] = folds
    seen = bytearray(len(y_by_id))
    groups: dict[str, int] = {}
    rows = 0
    with (dev_root / "oof_predictions.csv").open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["record_id", "origin_program", "y_true", "fold"]:
            raise ValueError("Development reference CSV schema is not exact")
        for row in reader:
            rid = int(row["record_id"])
            if rid >= len(seen) or seen[rid] or y_by_id[rid] != int(row["y_true"]) or fold_by_id[rid] != int(row["fold"]):
                raise ValueError(f"Development row differs from numeric reference arrays: {rid}")
            seen[rid] = 1
            origin, fold = row["origin_program"], int(row["fold"])
            previous = groups.setdefault(origin, fold)
            if not origin or previous != fold:
                raise ValueError(f"Origin crosses reference folds: {origin}")
            rows += 1
    if rows != EXPECTED_ROWS or len(groups) != EXPECTED_ORIGINS or sum(seen) != EXPECTED_ROWS:
        raise ValueError("Installed development reference CSV has invalid rows or groups")
    return {"source_commit": commit, "source_sha256": source_sha,
            "development_records": rows, "development_origins": len(groups),
            "reference_archive_sha256": sha256(archive), "frozen_test_rows": 0}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--official-checkout", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    print(json.dumps(install_inputs(args.source_csv, args.official_checkout, args.bundle, args.root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
