"""Verify existing full QMutBench development QASM feature caches without retraining.

Uses only metadata, the saved development OOF identifiers/labels, cache manifests,
and compressed feature chunks. It never opens a frozen-test artifact.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import subprocess
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/experiments/full_data_qml_optimization_v3"
OFFICIAL = ROOT / "data/qmutbench_full"
FULL_META = OFFICIAL / "data/all_data.csv"
INGESTION_META = ROOT / "data/qmutbench_ingestion/raw/official_all_data_20261015.csv"
OOF = ROOT / "data/qmutbench_ml/master_optimization_v1/oof_predictions.csv"
FEATURE_DIR = ROOT / "data/qmutbench_ml/qasm_structural_v4"
DIFF_DIR = ROOT / "data/qmutbench_ml/qasm_mutation_diff_v1"
EXPECTED_META_SHA = "3f4547868673c583f047cee19d27c7bc97c64f15bb0586098ada778a3a0ff778"
EXPECTED_OOF_SHA = "7d203ac4cd6878a1caaf19eb768bb9a1e4f322b42fc56475cdc603591cc250ce"
EXPECTED_OFFICIAL_COMMIT = "01f5e7700c051fe715fc368b307deaa7e6869ba8"
N_IDS = 723079


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def compare_metadata() -> dict:
    differing_rows = 0
    differing_fields: Counter[str] = Counter()
    examples = []
    scanned = 0
    with FULL_META.open("r", encoding="utf-8-sig", newline="") as af, INGESTION_META.open("r", encoding="utf-8-sig", newline="") as bf:
        a, b = csv.reader(af), csv.reader(bf)
        header_a, header_b = next(a), next(b)
        if header_a != header_b:
            return {"status": "SCHEMA_MISMATCH", "full_header": header_a, "ingestion_header": header_b}
        for row_a, row_b in zip(a, b):
            scanned += 1
            if row_a != row_b:
                differing_rows += 1
                for i, (x, y) in enumerate(zip(row_a, row_b)):
                    if x != y:
                        differing_fields[header_a[i] if i < len(header_a) else str(i)] += 1
                if len(examples) < 3:
                    examples.append({"row_index": scanned - 1, "checkout": row_a[:4], "ingestion": row_b[:4]})
        extra_a = next(a, None)
        extra_b = next(b, None)
    return {"rows_compared": scanned, "row_value_differences": differing_rows,
            "differing_field_counts": dict(differing_fields), "examples": examples,
            "extra_checkout_row": extra_a, "extra_ingestion_row": extra_b,
            "status": "VALUE_IDENTICAL" if differing_rows == 0 and extra_a is None and extra_b is None else "CONTENT_DIFFERS"}


def verify_chunks(cache_dir: Path, manifest_name: str, record_ids: bytearray, labels_by_id: bytearray | None = None) -> dict:
    manifest_path = cache_dir / manifest_name
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunks = manifest.get("chunks", [])
    total = 0
    errors_manifest = 0
    missing = 0
    duplicates = 0
    out_of_development = 0
    bad_nonfinite = 0
    match_counts: Counter[tuple[int, int]] = Counter()
    schema = None
    chunk_results = []
    seen = bytearray(N_IDS)
    for entry in chunks:
        fp = cache_dir / entry["file"]
        errp = cache_dir / entry["error_file"]
        errors_manifest += int(entry.get("errors", 0))
        if not fp.is_file() or not errp.is_file():
            chunk_results.append({"file": entry["file"], "status": "MISSING_FILE"})
            missing += 1
            continue
        digest = sha(fp)
        sha_ok = digest == entry.get("sha256")
        count = 0
        with gzip.open(fp, "rt", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames or []
            if schema is None:
                schema = fields
            for row in reader:
                count += 1
                total += 1
                try:
                    rid = int(row.get("record_id", ""))
                except (TypeError, ValueError):
                    out_of_development += 1
                    continue
                if rid < 0 or rid >= N_IDS or not record_ids[rid]:
                    out_of_development += 1
                elif seen[rid]:
                    duplicates += 1
                else:
                    seen[rid] = 1
                for name, value in row.items():
                    if name in {"record_id", "original_qasm_path", "mutant_qasm_path"}:
                        continue
                    try:
                        if not math.isfinite(float(value)):
                            bad_nonfinite += 1
                    except (TypeError, ValueError):
                        bad_nonfinite += 1
                if "source_gate_count_match" in row and labels_by_id is not None and 0 <= rid < N_IDS:
                    try:
                        match_counts[(int(labels_by_id[rid]), int(row["source_gate_count_match"]))] += 1
                    except (ValueError, TypeError):
                        bad_nonfinite += 1
        error_bytes = errp.stat().st_size
        chunk_ok = sha_ok and count == entry.get("rows") and error_bytes == 0
        chunk_results.append({"file": entry["file"], "rows_manifest": entry.get("rows"), "rows_actual": count,
                              "sha256_manifest": entry.get("sha256"), "sha256_actual": digest,
                              "sha256_matches": sha_ok, "error_log_bytes": error_bytes,
                              "status": "VERIFIED" if chunk_ok else "MISMATCH"})
    return {"manifest": str(manifest_path.relative_to(ROOT)).replace("\\", "/"),
            "manifest_status": manifest.get("status"), "manifest_records_requested": manifest.get("records_requested"),
            "manifest_feature_rows": manifest.get("feature_rows"), "manifest_parse_error_rows": manifest.get("parse_error_rows"),
            "manifest_chunks": len(chunks), "actual_feature_rows": total, "manifest_error_total": errors_manifest,
            "actual_unique_ids": sum(seen), "duplicate_ids": duplicates, "non_development_ids": out_of_development,
            "nonfinite_or_invalid_numeric_values": bad_nonfinite, "missing_chunks_or_logs": missing,
            "feature_schema": schema, "required_schema": manifest.get("feature_columns", []),
            "schema_required_columns_present": bool(schema) and set(manifest.get("feature_columns", [])) <= set(schema),
            "extra_non_feature_columns": sorted(set(schema or []) - set(manifest.get("feature_columns", [])) -
                                                  {"record_id", "original_qasm_path", "mutant_qasm_path"}),
            "source_gate_count_match_by_binary_label_and_value": {f"label={k[0]},match={k[1]}": v for k, v in match_counts.items()},
            "chunk_integrity_all_verified": bool(chunks) and all(c["status"] == "VERIFIED" for c in chunk_results),
            "chunk_results": chunk_results}


def main() -> int:
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    official_commit = subprocess.check_output(["git", "-C", str(OFFICIAL), "rev-parse", "HEAD"], text=True).strip()
    official_dirty = subprocess.check_output(["git", "-C", str(OFFICIAL), "status", "--porcelain"], text=True).strip()
    source_sha, oof_sha = sha(INGESTION_META), sha(OOF)
    raw_labels: Counter[str] = Counter()
    dev_labels: Counter[str] = Counter()
    fold_records: Counter[str] = Counter()
    fold_groups: dict[str, set[str]] = {}
    group_folds: dict[str, set[str]] = {}
    ids_seen = bytearray(N_IDS)
    labels_by_id = bytearray([255]) * N_IDS
    oof_rows = 0
    invalid_oof_ids = duplicate_oof_ids = invalid_origins = 0
    oof_header = None
    with OOF.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f); oof_header = r.fieldnames or []
        for row in r:
            oof_rows += 1
            try:
                rid = int(row["record_id"]); y = int(row["y_true"])
                fold = str(row["fold"]); origin = str(row["origin_program"])
            except (KeyError, TypeError, ValueError):
                invalid_oof_ids += 1; continue
            if not (0 <= rid < N_IDS) or y not in (0, 1) or not origin:
                invalid_oof_ids += 1; continue
            if ids_seen[rid]: duplicate_oof_ids += 1
            ids_seen[rid] = 1; labels_by_id[rid] = y
            dev_labels[str(y)] += 1; fold_records[fold] += 1
            group_folds.setdefault(origin, set()).add(fold)
    for origin, folds in group_folds.items():
        fold_groups["|".join(sorted(folds))] = fold_groups.get("|".join(sorted(folds)), set()) | {origin}

    id_mismatch = 0
    mapping_match = Counter()
    metadata_rows = metadata_duplicates = bad_metadata_ids = 0
    metadata_seen = bytearray(N_IDS)
    with INGESTION_META.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        idcol = reader.fieldnames[0]
        for row in reader:
            metadata_rows += 1
            raw = str(row.get("Killed", "")); raw_labels[raw] += 1
            try:
                rid = int(row[idcol])
            except (TypeError, ValueError):
                bad_metadata_ids += 1; continue
            if not (0 <= rid < N_IDS):
                bad_metadata_ids += 1; continue
            if metadata_seen[rid]: metadata_duplicates += 1
            metadata_seen[rid] = 1
            if labels_by_id[rid] != 255:
                y = labels_by_id[rid]
                expected = 0 if raw == "Survivor" else 1 if raw in {"WOO", "OPO"} else None
                if expected is None:
                    bad_metadata_ids += 1
                    continue
                mapping_match[f"{raw}:expected={expected}:actual={y}"] += 1
                id_mismatch += int(y != expected)

    feature_audit = verify_chunks(FEATURE_DIR, "manifest.json", ids_seen, labels_by_id)
    diff_audit = verify_chunks(DIFF_DIR, "manifest.json", ids_seen)
    meta_comparison = compare_metadata()
    check = {
        "generated_unix": time.time(),
        "branch_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "official_checkout_commit": official_commit, "official_checkout_expected_commit": EXPECTED_OFFICIAL_COMMIT,
        "official_checkout_clean": not bool(official_dirty),
        "metadata": {"ingestion_sha256": source_sha, "expected_sha256": EXPECTED_META_SHA,
                     "checksum_matches": source_sha == EXPECTED_META_SHA, "rows": metadata_rows,
                     "raw_label_counts": dict(raw_labels), "unique_in_range_record_ids": sum(metadata_seen),
                     "duplicate_ids": metadata_duplicates, "invalid_ids": bad_metadata_ids,
                     "checkout_vs_ingestion_row_value_audit": meta_comparison},
        "development_oof": {"path": str(OOF.relative_to(ROOT)).replace("\\", "/"), "sha256": oof_sha,
                             "expected_sha256": EXPECTED_OOF_SHA, "checksum_matches": oof_sha == EXPECTED_OOF_SHA,
                             "columns": oof_header, "rows": oof_rows, "unique_record_ids": sum(ids_seen),
                             "duplicate_record_ids": duplicate_oof_ids, "invalid_records": invalid_oof_ids,
                             "origin_count": len(group_folds), "fold_record_counts": dict(fold_records),
                             "origins_by_fold_signature": {k: len(v) for k, v in fold_groups.items()},
                             "origins_crossing_folds": sum(len(v) > 1 for v in group_folds.values()),
                             "binary_y_counts": dict(dev_labels),
                             "binary_mapping_WOO_1_OPO_1_Survivor_0_counts": dict(mapping_match),
                             "label_mapping_mismatches": id_mismatch,
                             "invalid_metadata_records": bad_metadata_ids},
        "qasm_structural_features": feature_audit,
        "qasm_mutation_diff_features": diff_audit,
        "frozen_test": {"accessed": False, "outcomes_read": False,
                         "checksum_status": "not independently rehashed; no frozen-test artifact was opened in this audit"},
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    check["overall_integrity"] = all([
        check["official_checkout_commit"] == EXPECTED_OFFICIAL_COMMIT,
        check["official_checkout_clean"], check["metadata"]["checksum_matches"],
        check["development_oof"]["checksum_matches"], check["development_oof"]["rows"] == 578117,
        check["development_oof"]["unique_record_ids"] == 578117,
        check["development_oof"]["duplicate_record_ids"] == 0, check["development_oof"]["invalid_records"] == 0,
        check["development_oof"]["origin_count"] == 305,
        check["development_oof"]["origins_crossing_folds"] == 0,
        check["development_oof"]["label_mapping_mismatches"] == 0,
        check["metadata"]["checkout_vs_ingestion_row_value_audit"]["status"] == "VALUE_IDENTICAL",
        check["metadata"]["rows"] == N_IDS, check["metadata"]["unique_in_range_record_ids"] == N_IDS,
        check["metadata"]["duplicate_ids"] == 0, check["metadata"]["invalid_ids"] == 0,
        feature_audit["manifest_status"] == "complete", feature_audit["chunk_integrity_all_verified"],
        feature_audit["actual_feature_rows"] == 578117, feature_audit["actual_unique_ids"] == 578117,
        feature_audit["duplicate_ids"] == 0, feature_audit["non_development_ids"] == 0,
        feature_audit["manifest_error_total"] == 0, feature_audit["nonfinite_or_invalid_numeric_values"] == 0,
        diff_audit["manifest_status"] == "complete", diff_audit["chunk_integrity_all_verified"],
        diff_audit["actual_feature_rows"] == 578117, diff_audit["actual_unique_ids"] == 578117,
        diff_audit["duplicate_ids"] == 0, diff_audit["non_development_ids"] == 0,
        diff_audit["manifest_error_total"] == 0,
    ])
    write_json(OUT / "feature_extraction_audit.json", check)
    (OUT / "label_leakage_audit.json").write_text(json.dumps({
        "status": "PASS" if check["development_oof"]["label_mapping_mismatches"] == 0 else "FAIL",
        "binary_label_mapping_verified": {"WOO": 1, "OPO": 1, "Survivor": 0},
        "semantics": "WOO and OPO are killed mutants (oracle-detected and not oracle-detected respectively); Survivor is not killed. This project’s binary target is killed-vs-survived, not oracle-detected-vs-undetected.",
        "source_gate_count_match": {"included_in_manifest_model_feature_columns": False,
                                    "present_as_diagnostic_in_chunk_csv": "source_gate_count_match" in (feature_audit.get("feature_schema") or []),
                                    "counts_by_label_and_value": feature_audit.get("source_gate_count_match_by_binary_label_and_value", {}),
                                    "decision": "Not part of the registered feature list and excluded from comparison models. It varies by label; treat as suspicious until a same-fold with/without ablation and provenance review are complete. Association alone does not establish target leakage."},
        "identifier_exclusion": ["record_id", "Origin_program", "source QASM paths", "fold"],
        "group_isolation": {"origins": check["development_oof"]["origin_count"], "cross_fold_origins": check["development_oof"]["origins_crossing_folds"]},
        "frozen_test_accessed": False,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"overall_integrity": check["overall_integrity"],
                      "dev_rows": oof_rows, "origins": len(group_folds), "folds": dict(fold_records),
                      "metadata_comparison": meta_comparison["status"],
                      "feature_rows": feature_audit["actual_feature_rows"],
                      "feature_errors": feature_audit["manifest_error_total"],
                      "diff_rows": diff_audit["actual_feature_rows"], "diff_errors": diff_audit["manifest_error_total"],
                      "label_mismatches": id_mismatch, "source_gate_count_match": feature_audit.get("source_gate_count_match_by_binary_label_and_value"),
                      "elapsed_seconds": check["elapsed_seconds"]}, indent=2))
    return 0 if check["overall_integrity"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
