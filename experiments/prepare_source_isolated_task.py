"""Stream, validate, and persist one source/task table without mixing sources.

The QSimBench pilot reuses its already recorded group folds. Input files are
read-only; processed partitions and manifests are written under data/processed.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
PILOT_INPUT = ROOT / "reports/experiments/qmutbench_multisource_qml_campaign_20261028/qsimbench_backend_fingerprint/normalized_manifest.csv"
PILOT_FOLDS = ROOT / "reports/experiments/qmutbench_multisource_qml_campaign_20261028/qsimbench_backend_fingerprint/resource_light_qnn/oof_predictions.csv"
PILOT_OUT = ROOT / "data/processed/qsimbench/backend_fingerprint_v3"
PILOT_CONTRACT = {
    "source_id": "qsimbench", "task_id": "backend_fingerprint",
    "target_semantics": "post-execution Aer vs fake_kyiv backend fingerprint; not fault prediction",
    "required": ["record_id", "bug_id", "label", "code", "source", "algorithm", "size"],
    "record_id_column": "record_id", "group_column": "bug_id", "label_column": "label", "feature_column": "code",
    "allowed_labels": ["0", "1"], "fold_record_column": "record_id", "fold_group_column": "group_id",
    "fold_column": "fold", "fold_model_column": "model", "fold_model_value": "amplitude_hybrid_3q_depth2",
    "source_label_column": "source", "source_label_map": {"aer_simulator": "0", "fake_kyiv": "1"},
    "required_values": {},
    "positive_integer_columns": ["size"],
    "json_numeric_ranges": {"histogram_features": {"entropy": [0, 1], "max_probability": [0, 1],
        "support_fraction": [0, 1], "mean_hamming_weight": [0, 1], "hamming_variance": [0, 1],
        "all_ones_probability": [0, 1], "bits": [1, 1000000], "shots": [1, 1000000000000]}},
    "input_path": "reports/experiments/qmutbench_multisource_qml_campaign_20261028/qsimbench_backend_fingerprint/normalized_manifest.csv",
    "fold_path": "reports/experiments/qmutbench_multisource_qml_campaign_20261028/qsimbench_backend_fingerprint/resource_light_qnn/oof_predictions.csv",
    "output_path": "data/processed/qsimbench/backend_fingerprint_v3",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    """Keep repository paths portable while allowing isolated temp fixtures."""
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def memory_snapshot() -> dict:
    try:
        import psutil
        process = psutil.Process()
        return {"available_mib": round(psutil.virtual_memory().available / 2**20, 1),
                "process_rss_mib": round(process.memory_info().rss / 2**20, 1)}
    except Exception:
        return {"available_mib": None, "process_rss_mib": None}


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def atomic_csv(path: Path, fields: list[str], rows: Iterable[dict]) -> str:
    import io
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    data = buffer.getvalue().encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    if path.exists() and sha256(path) == digest:
        return digest
    atomic_bytes(path, data)
    return digest


def prepare(input_path: Path, fold_path: Path, output: Path, chunk_size: int = 64,
            contract: dict | None = None) -> dict:
    """Validate one task's declared schema/labels/groups and persist chunks."""
    contract = contract or {}
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    started = time.monotonic()
    memory_start = memory_snapshot()
    if memory_start["available_mib"] is not None and memory_start["available_mib"] < 256:
        raise RuntimeError(f"Preprocessing memory floor not met: {memory_start['available_mib']} MiB available; 256 MiB required")
    source_hash, fold_hash = sha256(input_path), sha256(fold_path)
    feature_store_audit = None
    if contract.get("feature_store_manifest"):
        feature_manifest_path = ROOT / contract["feature_store_manifest"]
        feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
        feature_store_audit = {"path": display_path(feature_manifest_path),
                               "manifest_sha256": sha256(feature_manifest_path),
                               "status": feature_manifest.get("status"),
                               "feature_rows": feature_manifest.get("feature_rows"),
                               "error_rows": feature_manifest.get("error_rows"),
                               "source_sha256": feature_manifest.get("source_sha256"),
                               "qasm_commit": feature_manifest.get("qasm_commit"),
                               "chunk_rows": sum(int(part.get("rows", 0)) for part in feature_manifest.get("chunks", [])),
                               "chunks": len(feature_manifest.get("chunks", []))}
        if (feature_store_audit["status"] != "complete" or feature_store_audit["error_rows"] != 0 or
                feature_store_audit["feature_rows"] != contract.get("feature_rows_expected") or
                feature_store_audit["chunk_rows"] != feature_store_audit["feature_rows"] or
                feature_store_audit["source_sha256"] != contract.get("feature_source_sha256")):
            raise ValueError("External feature store manifest does not match the task contract")
    contract = contract or PILOT_CONTRACT
    config = {**contract,
              "fold_manifest": display_path(fold_path), "chunk_size": chunk_size,
              "contract_version": 1}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    prior_path = output / "manifest.json"
    progress_path = output / "progress.json"
    prior_progress = None
    if progress_path.exists():
        prior_progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if (prior_progress.get("input_sha256") != source_hash or
                prior_progress.get("fold_sha256") != fold_hash or
                prior_progress.get("config_sha256") != config_hash):
            raise RuntimeError("Existing preprocessing progress is incompatible; refusing unsafe resume")
        for item in prior_progress.get("partitions", []):
            path = ROOT / item["path"]
            if not path.is_file() or sha256(path) != item["sha256"]:
                raise RuntimeError(f"Preprocessing progress partition failed checksum verification: {item['path']}")
    if prior_path.exists():
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        if prior.get("input_sha256") != source_hash or prior.get("fold_sha256") != fold_hash or prior.get("config_sha256") != config_hash:
            raise RuntimeError("Existing processed manifest is incompatible; refusing unsafe overwrite/resume")
        outputs = [(item["path"], item["sha256"]) for item in prior.get("partitions", [])]
        outputs += [(prior.get("rejection_references", {}).get("path"), prior.get("rejection_references", {}).get("sha256")),
                    (prior.get("split_manifest", {}).get("path"), prior.get("split_manifest", {}).get("sha256"))]
        for rel, expected in outputs:
            path = ROOT / rel if rel else None
            if not path or not path.is_file() or sha256(path) != expected:
                raise RuntimeError(f"Existing processed output failed checksum verification: {rel}; refusing unsafe resume")
    rid_col = contract["record_id_column"]
    group_col = contract["group_column"]
    label_col = contract["label_column"]
    feature_col = contract.get("feature_column")
    folds, fold_groups, fold_targets = {}, {}, {}
    fold_opener = gzip.open if contract.get("fold_gzip") else open
    with fold_opener(fold_path, "rt", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if contract.get("fold_model_value") is not None and row.get(contract["fold_model_column"]) != contract["fold_model_value"]:
                continue
            rid = row[contract["fold_record_column"]]
            fold = int(row[contract["fold_column"]])
            if rid in folds and folds[rid] != fold:
                raise ValueError(f"conflicting recorded folds for {rid}")
            folds[rid] = fold
            fold_groups[rid] = str(row.get(contract.get("fold_group_column", ""), ""))
            target_col = contract.get("fold_target_column")
            if target_col:
                fold_targets[rid] = str(row.get(target_col, ""))
    if not folds:
        raise ValueError("no existing pilot OOF group-fold assignments found")

    counts = Counter()
    labels = Counter()
    duplicate_ids = Counter()
    reject_counts = Counter()
    group_folds: dict[str, int] = {}
    cross_fold_group = False
    chunks = list(prior_progress.get("partitions", [])) if prior_progress else []
    rejects, split_rows = [], []
    buffer = []
    input_ids = set()
    chunk_number = 0
    input_path = input_path.resolve()
    with input_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        required = config["required"]
        if not set(required).issubset(fields):
            raise ValueError(f"missing columns: {sorted(set(required) - set(fields))}")
        def flush(index: int) -> None:
            nonlocal chunk_number
            if not buffer:
                return
            chunk_number += 1
            target = output / "cleaned" / f"part-{chunk_number:05d}.csv"
            digest = atomic_csv(target, fields, buffer)
            record = {"partition_id": target.stem, "path": display_path(target),
                      "rows": len(buffer), "sha256": digest}
            if chunk_number <= len(chunks):
                if chunks[chunk_number - 1] != record:
                    raise RuntimeError(f"Reprocessed partition differs from verified checkpoint: {target.name}")
            else:
                chunks.append(record)
            atomic_bytes(progress_path, (json.dumps({"status": "IN_PROGRESS", "input_sha256": source_hash,
                          "fold_sha256": fold_hash, "config_sha256": config_hash,
                          "partitions": chunks}, indent=2, sort_keys=True) + "\n").encode())
            buffer.clear()
        for line_no, row in enumerate(reader, 2):
            counts["input"] += 1
            rid, group = (row.get(rid_col) or "").strip(), (row.get(group_col) or "").strip()
            duplicate_in_input = bool(rid and rid in input_ids)
            if rid:
                input_ids.add(rid)
            label = (row.get(label_col) or "").strip()
            reasons = []
            if not rid: reasons.append("missing_record_id")
            elif duplicate_in_input: reasons.append("duplicate_record_id")
            if not group: reasons.append("missing_group_id")
            if label not in set(contract["allowed_labels"]): reasons.append("invalid_label")
            source_col = contract.get("source_label_column")
            if source_col:
                source_value = (row.get(source_col) or "").strip()
                expected = contract.get("source_label_map", {}).get(source_value)
                if expected is None: reasons.append("unsupported_source_label")
                elif label != str(expected): reasons.append("inconsistent_source_label")
            if feature_col and not (row.get(feature_col) or "").strip(): reasons.append("missing_feature_text")
            for col in contract.get("positive_integer_columns", []):
                try:
                    if int(row.get(col, "")) < 1: raise ValueError
                except (ValueError, TypeError):
                    reasons.append(f"invalid_positive_integer:{col}")
            for col, bounds in contract.get("numeric_ranges", {}).items():
                try:
                    value = float(row.get(col, ""))
                    if not math.isfinite(value) or not float(bounds[0]) <= value <= float(bounds[1]):
                        reasons.append(f"numeric_out_of_range:{col}")
                except (ValueError, TypeError):
                    reasons.append(f"invalid_numeric:{col}")
            for col, expected in contract.get("required_values", {}).items():
                if (row.get(col) or "").strip() != str(expected):
                    reasons.append(f"unexpected_value:{col}")
            rule = contract.get("derived_label_rule")
            if rule:
                try:
                    numeric = float(row.get(rule["column"], ""))
                    expected_label = str(rule["positive"] if numeric >= float(rule["threshold"]) else rule["negative"])
                    if label != expected_label:
                        reasons.append(f"derived_label_mismatch:{label_col}:{rule['column']}")
                except (ValueError, TypeError):
                    reasons.append(f"invalid_derived_label_feature:{rule['column']}")
            for col, ranges in contract.get("json_numeric_ranges", {}).items():
                try:
                    values = json.loads(row.get(col, ""))
                    for key, bounds in ranges.items():
                        value = float(values[key])
                        if not math.isfinite(value) or not float(bounds[0]) <= value <= float(bounds[1]):
                            reasons.append(f"numeric_out_of_range:{col}.{key}")
                except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                    reasons.append(f"invalid_numeric_json:{col}")
            if rid and rid not in folds: reasons.append("missing_recorded_fold")
            label_map = contract.get("fold_label_map")
            if rid in folds and label_map is not None and str(label_map.get(label, "")) != fold_targets.get(rid, ""):
                reasons.append("fold_target_mismatch")
            expected_group = fold_groups.get(rid, "")
            if expected_group and group and expected_group != group: reasons.append("fold_group_mismatch")
            if rid in folds and group and group in group_folds and group_folds[group] != folds[rid]:
                cross_fold_group = True
                reasons.append("group_crosses_folds")
            if reasons:
                if "duplicate_record_id" in reasons:
                    duplicate_ids[rid] += 1
                rejects.append({"input_line": line_no, "record_id": rid, "reason_codes": ";".join(reasons)})
                reject_counts.update(reasons)
                continue
            group_folds[group] = folds[rid]
            row[rid_col], row[group_col] = rid, group
            if feature_col:
                row[feature_col] = row[feature_col].strip()
            buffer.append(row)
            counts["accepted"] += 1
            labels[label] += 1
            role = "grouped_oof_validation"
            source_col = contract.get("training_source_column")
            if source_col:
                role = ("in_domain_grouped_oof_validation" if row.get(source_col) == contract.get("training_source_value")
                        else "transfer_target_evaluation")
            split_rows.append({"record_id": rid, "group_id": group, "fold": folds[rid], "role": role})
            if len(buffer) == chunk_size:
                flush(len(chunks) + 1)
        flush(len(chunks) + 1)
    if set(folds) != input_ids:
        missing = sorted(set(folds) - input_ids)
        counts["fold_records_not_in_input"] = len(missing)
        for rid in missing:
            rejects.append({"input_line": "", "record_id": rid, "reason_codes": "fold_record_not_in_input"})
            reject_counts["fold_record_not_in_input"] += 1
    reject_path = output / "rejected" / "rejection_references.csv"
    reject_digest = atomic_csv(reject_path, ["input_line", "record_id", "reason_codes"], rejects)
    split_path = output / "manifests" / "group_folds.csv"
    split_digest = atomic_csv(split_path, ["record_id", "group_id", "fold", "role"], sorted(split_rows, key=lambda x: x["record_id"]))
    source_groups = {}
    for row in split_rows:
        source_groups.setdefault(row["group_id"], row["fold"])
    cross_fold = cross_fold_group
    elapsed = round(time.monotonic() - started, 3)
    memory_end = memory_snapshot()
    report = {"status": "VALIDATED" if not rejects and counts["accepted"] == counts["input"] else "VALIDATED_WITH_REJECTIONS",
              "source_id": contract["source_id"], "task_id": contract["task_id"],
              "target_semantics": contract["target_semantics"],
              "input_path": display_path(input_path), "input_sha256": source_hash,
              "fold_input_path": display_path(fold_path), "fold_sha256": fold_hash,
              "feature_store": feature_store_audit,
              "config": config, "config_sha256": config_hash, "counts": dict(counts),
              "class_counts": dict(labels), "rejection_reason_counts": dict(reject_counts),
              "duplicate_record_ids": duplicate_ids.total(), "groups": len(source_groups),
              "fold_count": len(set(group_folds.values())), "group_cross_fold": cross_fold,
              "partitions": chunks, "rejection_references": {"path": display_path(reject_path), "sha256": reject_digest, "rows": len(rejects)},
              "split_manifest": {"path": display_path(split_path), "sha256": split_digest, "rows": len(split_rows)},
              "elapsed_seconds": elapsed, "input_rows_per_second": round(counts["input"] / max(elapsed, 1e-9), 2),
              "memory_floor_mib": 256,
              "memory_telemetry": {"start": memory_start, "end": memory_end,
                                   "peak_sampled_process_rss_mib": max(x for x in (
                                       memory_start["process_rss_mib"], memory_end["process_rss_mib"]) if x is not None)
                                   if any(x is not None for x in (memory_start["process_rss_mib"], memory_end["process_rss_mib"])) else None},
              "chunk_size": chunk_size,
              "preprocessing": "schema/label/group validation, deterministic whitespace trim only; no learned transformation",
              "limitations": (["uses task-specific saved group-fold manifest", "input is a derived task manifest; raw source remains untouched"]
                              if fold_path.name == "group_folds.csv" else
                              ["reuses fold assignment recorded in existing QNN OOF predictions", "input is an existing derived task manifest; raw source remains untouched"])}
    report["validation_pass"] = (not rejects and counts["accepted"] == counts["input"] and
                                 len(folds) == counts["accepted"] and not cross_fold)
    if not report["validation_pass"]:
        report["status"] = "VALIDATION_FAILED_OR_REJECTED"
    atomic_bytes(prior_path, (json.dumps(report, indent=2, sort_keys=True) + "\n").encode())
    atomic_bytes(progress_path, (json.dumps({"status": "COMPLETED", "input_sha256": source_hash,
                  "fold_sha256": fold_hash, "config_sha256": config_hash,
                  "partitions": chunks}, indent=2, sort_keys=True) + "\n").encode())
    md = ["# QSimBench backend fingerprint preprocessing pilot", "", f"Status: **{report['status']}**", "",
          "This source-specific task predicts backend identity from post-execution output descriptors. It is not hardware fault detection or localization.", "",
          f"- Rows: {counts['input']} input, {counts['accepted']} accepted, {len(rejects)} rejection references.",
          f"- Labels: `{dict(labels)}`; groups: {len(source_groups)}; folds: {len(set(group_folds.values()))}; cross-fold groups: {cross_fold}.",
          f"- Input SHA-256: `{source_hash}`; fold source SHA-256: `{fold_hash}`.",
          f"- Chunk size: {chunk_size}; partitions: {len(chunks)}; elapsed: {report['elapsed_seconds']} seconds; throughput: {report['input_rows_per_second']} rows/s.",
          f"- Memory floor: {report['memory_floor_mib']} MiB; available RAM at start/end: {memory_start['available_mib']}/{memory_end['available_mib']} MiB; sampled process RSS: {report['memory_telemetry']['peak_sampled_process_rss_mib']} MiB.",
          "- Learned preprocessing: none; split assignments reused from the existing QNN OOF manifest.",
          "- Raw source and historical training artifacts were not modified; no training job was started.", "",
          "Resume or reproduce with: `python experiments/prepare_source_isolated_task.py --chunk-size 64`. ",
          "The script verifies input, fold, and contract hashes before reusing an existing output manifest.", ""]
    atomic_bytes(output / "report.md", "\n".join(md).encode())
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=PILOT_INPUT)
    parser.add_argument("--folds", type=Path, default=PILOT_FOLDS)
    parser.add_argument("--output", type=Path, default=PILOT_OUT)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--contract", type=Path)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8")) if args.contract else None
    if contract:
        args.input = ROOT / contract["input_path"]
        args.folds = ROOT / contract["fold_path"]
        args.output = ROOT / contract["output_path"]
    result = prepare(args.input, args.folds, args.output, args.chunk_size, contract)
    print(json.dumps(result, indent=2))
    if not result["validation_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
