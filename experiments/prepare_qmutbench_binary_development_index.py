"""Build QMutBench's existing SQLite development index under a RAM gate.

DevelopmentStream verifies the official source, saved development split, target
mapping, and complete QASM cache. Frozen rows are never retained or label-read.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "data/processed/qmutbench/binary_development"
FINAL_INDEX = OUT / "reference_index.sqlite"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def main() -> None:
    import psutil
    available = round(psutil.virtual_memory().available / 2**20, 1)
    floor = 512
    if available < floor:
        raise MemoryError(f"QMutBench index preflight: {available} MiB available; {floor} MiB required")
    if FINAL_INDEX.exists():
        prior_path = OUT / "manifest.json"
        if not prior_path.is_file():
            raise FileExistsError("QMutBench index exists without a manifest; refusing to overwrite")
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        if sha256(FINAL_INDEX) != prior.get("index_sha256"):
            raise ValueError("Existing QMutBench SQLite index checksum mismatch")
        print(json.dumps({"status": "already_validated", "manifest": str(prior_path.relative_to(ROOT)),
                          "development_rows": prior["development_rows"]}, indent=2))
        return

    from qmutbench_qml.data.streaming import DevelopmentStream
    OUT.mkdir(parents=True, exist_ok=True)
    temp_index = OUT / f"reference_index.building.{uuid.uuid4().hex}.sqlite"
    stream = DevelopmentStream(ROOT, temp_index, metadata_chunk_rows=2000, feature_chunk_rows=2048,
                               archive_authoritative=True)
    started = time.perf_counter()
    rss_start = round(psutil.Process().memory_info().rss / 2**20, 1)
    try:
        audit = stream.build_index()
        final_available = round(psutil.virtual_memory().available / 2**20, 1)
        if final_available < floor:
            raise MemoryError(f"QMutBench index finished below RAM floor: {final_available} MiB")
        os.replace(temp_index, FINAL_INDEX)
        report = {"status": "READY_FOR_DEVELOPMENT_TRAINING", "source_id": "qmutbench",
                  "task_id": "binary_mutation_outcome", "preprocessing_status": "VALIDATED",
                  "training_status": "BLOCKED_MEMORY_GUARD",
                  "target_semantics": "Survivor=0; WOO/OPO=1 (historical mutation execution outcome, not hardware anomaly/bug attribution)",
                  "source_version": "official QMutBench ingestion 2026-10-15; QASM commit 01f5e7700c051fe715fc368b307deaa7e6869ba8",
                  "index_path": str(FINAL_INDEX.relative_to(ROOT)), "index_sha256": sha256(FINAL_INDEX),
                  "source_audit": audit, "elapsed_seconds": round(time.perf_counter() - started, 3),
                  "memory_floor_mib": floor, "available_ram_start_mib": available,
                  "available_ram_end_mib": final_available, "process_rss_start_mib": rss_start,
                  "process_rss_end_mib": round(psutil.Process().memory_info().rss / 2**20, 1),
                  "metadata_chunk_rows": 2000, "qasm_feature_chunk_rows": 2048,
                  "frozen_test_accessed": False,
                  "frozen_test_policy": "reference rows are allowlisted from development OOF; streaming adapter does not read frozen labels or predictors",
                  "next_action": "Train only after the existing 2 GiB startup preflight and 1 GiB in-run floor pass.",
                  "resumption": "Completed index can be reused after checksum validation. If interrupted before final rename, rerun to build a fresh uniquely named temporary SQLite index."}
        atomic_json(OUT / "manifest.json", report)
        md = ["# QMutBench binary development preprocessing", "",
              f"Status: **{report['status']}**", "",
              "This is historical mutation-outcome classification (Survivor versus WOO/OPO); it is not hardware fault diagnosis.", "",
              f"- Source SHA-256: `{audit['source_sha256']}`; reference SHA-256: `{audit['reference_sha256']}`.",
              f"- Development: {audit['development_rows_indexed']:,} rows / {audit['development_origins']} origin groups; frozen rows retained: {audit['frozen_test_rows_retained']}.",
              f"- Class counts: `{audit['class_counts']}`; grouped fold origin counts: `{audit['fold_origin_counts']}`.",
              f"- QASM feature cache SHA-256: `{audit['qasm_manifest_sha256']}`; cache rows verified: {audit['development_rows_indexed']:,}.",
              f"- Chunk sizes: metadata 2,000; QASM cache 2,048; elapsed {report['elapsed_seconds']} s.",
              f"- Available RAM start/end: {available}/{final_available} MiB; process RSS start/end: {rss_start}/{report['process_rss_end_mib']} MiB.",
              "- Preprocessing is ready; QNN training remains blocked by the established 2 GiB startup / 1 GiB in-run memory guards.",
              "- Frozen test data were not read for this task.", "",
              "Re-run: `.venv\\Scripts\\python.exe experiments\\prepare_qmutbench_binary_development_index.py`.", ""]
        (OUT / "report.md").write_text("\n".join(md), encoding="utf-8")
        print(json.dumps({"status": report["status"], "development_rows": audit["development_rows_indexed"],
                          "groups": audit["development_origins"], "index_sha256": report["index_sha256"],
                          "elapsed_seconds": report["elapsed_seconds"], "memory_end_mib": final_available}, indent=2))
    except Exception:
        failure = {"status": "INTERRUPTED_OR_FAILED", "temporary_index": str(temp_index.relative_to(ROOT)),
                   "source_version": "official QMutBench ingestion 2026-10-15",
                   "frozen_test_accessed": False, "error": "see command stderr; temporary index is not eligible for training"}
        atomic_json(OUT / "failed_attempt.json", failure)
        raise


if __name__ == "__main__":
    main()
