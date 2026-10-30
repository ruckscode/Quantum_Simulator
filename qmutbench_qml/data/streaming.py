"""Disk-indexed, bounded-memory reader for development-only QMutBench rows."""
from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Iterator

EXPECTED_SOURCE_SHA256 = "3f4547868673c583f047cee19d27c7bc97c64f15bb0586098ada778a3a0ff778"
EXPECTED_QASM_COMMIT = "01f5e7700c051fe715fc368b307deaa7e6869ba8"
EXPECTED_REFERENCE_ROWS = 578_117
EXPECTED_REFERENCE_ORIGINS = 305
METADATA_FIELDS = ["algorithm", "operation", "gate", "dominant_state", "qubits", "gates",
                   "Position_int", "position"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class DevelopmentStream:
    """Build a SQLite disk index, then join only bounded QASM feature chunks."""

    def __init__(self, root: Path, index_path: Path, *, expected_rows: int = EXPECTED_REFERENCE_ROWS,
                 expected_origins: int = EXPECTED_REFERENCE_ORIGINS,
                 expected_source_rows: int = 723_079,
                 expected_source_sha256: str | None = EXPECTED_SOURCE_SHA256,
                 metadata_chunk_rows: int = 10_000, feature_chunk_rows: int = 10_000,
                 reference_csv_path: Path | None = None,
                 reference_archive_path: Path | None = None,
                 archive_authoritative: bool = False):
        if metadata_chunk_rows < 1 or feature_chunk_rows < 1:
            raise ValueError("stream chunk sizes must be positive")
        self.root, self.index_path = Path(root), Path(index_path)
        self.expected_rows, self.expected_origins = expected_rows, expected_origins
        self.expected_source_rows = expected_source_rows
        self.expected_source_sha256 = expected_source_sha256
        self.metadata_chunk_rows, self.feature_chunk_rows = metadata_chunk_rows, feature_chunk_rows
        self.raw_path = self.root / "data/qmutbench_ingestion/raw/official_all_data_20261015.csv"
        self.reference_path = Path(reference_csv_path) if reference_csv_path else self.root / "data/qmutbench_ml/master_optimization_v1/oof_predictions.csv"
        self.archive_path = Path(reference_archive_path) if reference_archive_path else self.root / "data/qmutbench_ml/exhaustive_campaign_v1/five_fold_confirmation/mutation_diff_oof.npz"
        self.archive_authoritative = bool(archive_authoritative)
        self.cache_path = self.root / "data/qmutbench_ml/qasm_mutation_diff_v1"
        self.audit: dict = {}
        self.last_stream_audit: dict = {}
        required_paths = [self.raw_path, self.archive_path, self.cache_path / "manifest.json"]
        if not self.archive_authoritative:
            required_paths.append(self.reference_path)
        for path in required_paths:
            if not path.is_file():
                raise FileNotFoundError(f"Required QMutBench streaming input is missing: {path}")

    def build_index(self) -> dict:
        import numpy as np

        raw_sha = sha256(self.raw_path)
        if self.expected_source_sha256 and raw_sha != self.expected_source_sha256:
            raise ValueError(f"Source metadata checksum mismatch: {self.raw_path}")
        with np.load(self.archive_path, allow_pickle=False) as source_archive:
            size = int(source_archive["record_id"].max()) + 1
        y_by_id = np.full(size, -1, dtype=np.int8)
        fold_by_id = np.zeros(size, dtype=np.int8)
        origin_by_id = np.full(size, -1, dtype=np.int16)
        origin_names: list[str] = []
        origin_to_code: dict[str, int] = {}
        seen_reference = np.zeros(size, dtype=np.bool_)
        fold_origin: dict[str, int] = {}
        reference_count = 0
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        if self.index_path.exists():
            raise FileExistsError(f"Refusing to replace an existing streaming index: {self.index_path}")
        connection = sqlite3.connect(self.index_path)
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("CREATE TABLE reference (record_id INTEGER PRIMARY KEY, origin TEXT NOT NULL, y INTEGER NOT NULL, fold INTEGER NOT NULL)")
            if self.archive_authoritative:
                with np.load(self.archive_path, allow_pickle=False) as archive:
                    required = {"record_id", "y", "fold"}
                    if not required <= set(archive.files):
                        raise ValueError("Numeric reference archive lacks record_id/y/fold arrays")
                    ids = np.asarray(archive["record_id"], dtype=np.int64)
                    ay = np.asarray(archive["y"], dtype=np.int8)
                    af = np.asarray(archive["fold"], dtype=np.int8)
                if (ids.ndim != 1 or ay.shape != ids.shape or af.shape != ids.shape or
                        len(ids) != self.expected_rows or np.any(ids < 0) or np.any(ids >= size) or
                        len(np.unique(ids)) != len(ids) or not np.isin(ay, [0, 1]).all() or
                        not np.isin(af, [1, 2, 3, 4, 5]).all()):
                    raise ValueError("Authoritative development archive has invalid IDs, targets, or folds")
                y_by_id[ids], fold_by_id[ids], seen_reference[ids] = ay, af, True
                reference_count = len(ids)
            else:
                with self.reference_path.open("r", encoding="utf-8-sig", newline="") as stream:
                    reader = csv.DictReader(stream)
                    required_reference_fields = {"record_id", "origin_program", "y_true", "fold"}
                    if (not reader.fieldnames or not required_reference_fields <= set(reader.fieldnames) or
                            len(reader.fieldnames) != len(set(reader.fieldnames))):
                        raise ValueError("Development reference CSV lacks unique required fields")
                    batch = []
                    for row in reader:
                        rid, y, fold = int(row["record_id"]), int(row["y_true"]), int(row["fold"])
                        origin = row["origin_program"]
                        if rid < 0 or rid >= size or y not in (0, 1) or fold not in range(1, 6) or not origin:
                            raise ValueError(f"Invalid development reference row: {rid}")
                        if seen_reference[rid]:
                            raise ValueError(f"Duplicate development reference record ID: {rid}")
                        seen_reference[rid] = True
                        code = origin_to_code.setdefault(origin, len(origin_names))
                        if code == len(origin_names):
                            origin_names.append(origin)
                        if fold_origin.setdefault(origin, fold) != fold:
                            raise ValueError(f"Origin {origin!r} crosses reference folds")
                        y_by_id[rid], fold_by_id[rid], origin_by_id[rid] = y, fold, code
                        batch.append((rid, origin, y, fold))
                        if len(batch) >= self.metadata_chunk_rows:
                            connection.executemany("INSERT INTO reference VALUES (?, ?, ?, ?)", batch)
                            batch.clear()
                        reference_count += 1
                    if batch:
                        connection.executemany("INSERT INTO reference VALUES (?, ?, ?, ?)", batch)
                if reference_count != self.expected_rows or len(fold_origin) != self.expected_origins:
                    raise ValueError("Saved development reference row/origin counts differ from contract")

            with np.load(self.archive_path, allow_pickle=False) as archive:
                required = {"record_id", "y", "fold"}
                if not required <= set(archive.files):
                    raise ValueError("Numeric reference archive lacks record_id/y/fold arrays")
                ids = np.asarray(archive["record_id"], dtype=np.int64)
                ay = np.asarray(archive["y"], dtype=np.int8)
                af = np.asarray(archive["fold"], dtype=np.int8)
            if (ids.ndim != 1 or ay.shape != ids.shape or af.shape != ids.shape or
                    len(ids) != reference_count or np.any(ids < 0) or np.any(ids >= size) or
                    len(np.unique(ids)) != len(ids) or not np.isin(ay, [0, 1]).all() or
                    not np.isin(af, [1, 2, 3, 4, 5]).all() or
                    not np.array_equal(y_by_id[ids], ay) or not np.array_equal(fold_by_id[ids], af)):
                raise ValueError("Development split assignments disagree with numeric reference archive")

            columns = ", ".join(f'"{name}" TEXT' for name in METADATA_FIELDS)
            connection.execute(f"CREATE TABLE metadata (record_id INTEGER PRIMARY KEY, origin TEXT NOT NULL, y INTEGER NOT NULL, fold INTEGER NOT NULL, {columns})")
            with self.raw_path.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.reader(stream)
                header = next(reader)
                id_name = "" if header and header[0] == "" else "Unnamed: 0"
                indexes = {name: header.index(name) for name in [id_name, "Origin_program", "Killed", *METADATA_FIELDS]}
                id_index = indexes[id_name]
                selected = ["Origin_program", "Killed", *METADATA_FIELDS]
                insert_sql = "INSERT INTO metadata VALUES (" + ",".join("?" for _ in range(4 + len(METADATA_FIELDS))) + ")"
                batch = []
                reference_batch = []
                source_rows = development_rows = 0
                seen_source = np.zeros(size, dtype=np.bool_)
                for row in reader:
                    source_rows += 1
                    rid = int(row[id_index])
                    if rid < 0 or rid >= size or seen_source[rid]:
                        raise ValueError(f"Unexpected or duplicate source metadata record ID: {rid}")
                    seen_source[rid] = True
                    if not seen_reference[rid]:
                        continue  # Do not inspect labels, groups, or predictors for frozen-test rows.
                    if len(row) <= max(indexes.values()):
                        raise ValueError(f"Truncated metadata row for development record {rid}")
                    vals = {name: row[indexes[name]] for name in selected}
                    origin, label = vals["Origin_program"], vals["Killed"]
                    y = {"Survivor": 0, "WOO": 1, "OPO": 1}.get(label)
                    if y is None or y != int(y_by_id[rid]):
                        raise ValueError(f"Source/reference target mismatch for development record {rid}")
                    if self.archive_authoritative:
                        code = origin_to_code.setdefault(origin, len(origin_names))
                        if code == len(origin_names):
                            origin_names.append(origin)
                        fold = int(fold_by_id[rid])
                        if fold_origin.setdefault(origin, fold) != fold:
                            raise ValueError(f"Origin {origin!r} crosses reference folds")
                        origin_by_id[rid] = code
                        reference_batch.append((rid, origin, y, fold))
                        if len(reference_batch) >= self.metadata_chunk_rows:
                            connection.executemany("INSERT INTO reference VALUES (?, ?, ?, ?)", reference_batch)
                            reference_batch.clear()
                    else:
                        expected_origin = origin_names[int(origin_by_id[rid])]
                        if origin != expected_origin:
                            raise ValueError(f"Reference origin mismatch for development record {rid}")
                    batch.append((rid, origin, y, int(fold_by_id[rid]),
                                  *(vals[name] for name in METADATA_FIELDS)))
                    if len(batch) >= self.metadata_chunk_rows:
                        connection.executemany(insert_sql, batch)
                        batch.clear()
                    development_rows += 1
                if batch:
                    connection.executemany(insert_sql, batch)
                if reference_batch:
                    connection.executemany("INSERT INTO reference VALUES (?, ?, ?, ?)", reference_batch)
            if self.archive_authoritative and (reference_count != self.expected_rows or len(origin_names) != self.expected_origins):
                raise ValueError("Archive-authoritative development rows/origins differ from contract")
            if source_rows != self.expected_source_rows:
                raise ValueError(f"Source metadata row count {source_rows} differs from expected {self.expected_source_rows}")
            indexed = int(connection.execute("SELECT COUNT(*) FROM metadata").fetchone()[0])
            missing = int(connection.execute("SELECT COUNT(*) FROM reference r LEFT JOIN metadata m USING(record_id) WHERE m.record_id IS NULL").fetchone()[0])
            if indexed != self.expected_rows or development_rows != indexed or missing:
                raise ValueError(f"Development source reconciliation failed: indexed={indexed}, missing={missing}")
            connection.execute("CREATE INDEX metadata_fold_idx ON metadata(fold, record_id)")
            fold_class_counts = {str(fold): {str(y): int(connection.execute(
                "SELECT COUNT(*) FROM metadata WHERE fold=? AND y=?", (fold, y)).fetchone()[0])
                for y in (0, 1)} for fold in range(1, 6)}
            if (self.expected_rows == EXPECTED_REFERENCE_ROWS and
                    any(count == 0 for counts in fold_class_counts.values() for count in counts.values())):
                raise ValueError("Saved development fold lacks one of the binary classes")
            connection.commit()
            cache_manifest = json.loads((self.cache_path / "manifest.json").read_text(encoding="utf-8"))
            if (cache_manifest.get("qasm_commit") != EXPECTED_QASM_COMMIT or
                    cache_manifest.get("source_sha256") != raw_sha or
                    cache_manifest.get("feature_rows") != self.expected_rows or
                    cache_manifest.get("error_rows") != 0 or cache_manifest.get("status") != "complete"):
                raise ValueError("QASM mutation-diff cache is incomplete or has a different source")
            cached_rows = 0
            for part in cache_manifest.get("chunks", []):
                if int(part.get("errors", 0)) != 0:
                    raise ValueError("QASM mutation-diff cache contains feature errors")
                part_path = self.cache_path / part["file"]
                if sha256(part_path) != part.get("sha256"):
                    raise ValueError(f"QASM feature chunk checksum mismatch: {part_path}")
                cached_rows += int(part.get("rows", 0))
            if cached_rows != self.expected_rows:
                raise ValueError(f"QASM cache manifest covers {cached_rows}, expected {self.expected_rows} rows")
            self._origin_by_id, self._origin_names = origin_by_id, origin_names
            self._y_by_id, self._fold_by_id, self._seen_reference = y_by_id, fold_by_id, seen_reference
            self.audit = {"source_sha256": raw_sha,
                "reference_sha256": None if self.archive_authoritative else sha256(self.reference_path),
                "reference_authority": "numeric five-fold development archive" if self.archive_authoritative else "saved development CSV reconciled to numeric archive",
                "reference_archive_sha256": sha256(self.archive_path),
                "qasm_manifest_sha256": sha256(self.cache_path / "manifest.json"),
                "source_rows_scanned": source_rows, "development_rows_indexed": indexed,
                "development_origins": len(origin_names), "frozen_test_rows_retained": 0,
                "source_chunk_rows": self.metadata_chunk_rows,
                "fold_record_counts": {str(f): int(connection.execute("SELECT COUNT(*) FROM metadata WHERE fold=?", (f,)).fetchone()[0]) for f in range(1, 6)},
                "fold_origin_counts": {str(f): int(connection.execute("SELECT COUNT(DISTINCT origin) FROM metadata WHERE fold=?", (f,)).fetchone()[0]) for f in range(1, 6)},
                "fold_class_counts": fold_class_counts,
                "class_counts": {str(y): int(connection.execute("SELECT COUNT(*) FROM metadata WHERE y=?", (y,)).fetchone()[0]) for y in (0, 1)}}
            return self.audit
        finally:
            connection.close()

    def iter_fold(self, fold: int, role: str) -> Iterator:
        """Yield aligned pandas feature frames; role is training or validation."""
        import pandas as pd
        from qmutbench_qml.data.loader import FEATURES, QASM_CATEGORICAL, QASM_NUMERIC

        if role not in {"training", "validation"} or fold not in range(1, 6):
            raise ValueError("role/fold is invalid")
        if not self.audit:
            raise RuntimeError("Call build_index() before streaming feature chunks")
        con = sqlite3.connect(self.index_path)
        import numpy as np
        seen_cache = np.zeros_like(self._seen_reference)
        total = 0
        selected_seen = np.zeros_like(self._seen_reference)
        origins_seen, class_counts = set(), {0: 0, 1: 0}
        try:
            manifest = json.loads((self.cache_path / "manifest.json").read_text(encoding="utf-8"))
            for part_info in manifest["chunks"]:
                if int(part_info.get("errors", 0)) != 0:
                    raise ValueError("QASM feature cache contains extraction errors")
                path = self.cache_path / part_info["file"]
                if sha256(path) != part_info["sha256"]:
                    raise ValueError(f"QASM feature chunk checksum mismatch: {path}")
                part_rows = 0
                for qasm in pd.read_csv(path, chunksize=self.feature_chunk_rows):
                    part_rows += len(qasm)
                    ids = qasm.record_id.astype("int64").to_numpy()
                    if len(ids) != len(set(map(int, ids))):
                        raise ValueError("QASM feature chunk contains duplicate record IDs")
                    for rid in ids:
                        rid = int(rid)
                        if rid < 0 or rid >= len(self._seen_reference) or not self._seen_reference[rid] or seen_cache[rid]:
                            raise ValueError(f"Unexpected or duplicate QASM cache record ID: {rid}")
                        seen_cache[rid] = True
                    selected_mask = self._fold_by_id[ids] == (fold if role == "validation" else self._fold_by_id[ids])
                    if role == "training":
                        selected_mask = self._fold_by_id[ids] != fold
                    selected = qasm.loc[selected_mask].copy()
                    if selected.empty:
                        continue
                    selected_ids = selected.record_id.astype(int).tolist()
                    rows = []
                    for offset in range(0, len(selected_ids), 400):
                        batch = selected_ids[offset:offset + 400]
                        marks = ",".join("?" for _ in batch)
                        rows.extend(con.execute(
                            f"SELECT record_id, origin, y, fold, algorithm, operation, gate, dominant_state, qubits, gates, Position_int, position FROM metadata WHERE record_id IN ({marks})",
                            batch).fetchall())
                    meta = pd.DataFrame(rows, columns=["record_id", "origin_program", "y", "fold", *METADATA_FIELDS])
                    if len(meta) != len(selected) or meta.record_id.duplicated().any():
                        raise ValueError("Metadata/QASM chunk join is incomplete or duplicated")
                    frame = meta.merge(selected, on="record_id", validate="one_to_one")
                    if role == "validation" and (frame.fold != fold).any():
                        raise ValueError("Validation stream contains a different fold")
                    if role == "training" and (frame.fold == fold).any():
                        raise ValueError("Training stream contains held-out fold rows")
                    for name in ("qubits", "gates", "Position_int", *QASM_NUMERIC):
                        frame[name] = pd.to_numeric(frame[name], errors="coerce")
                    frame["position_fraction"] = frame.position.map({
                        "0-10%": .05, "11-20%": .155, "21-30%": .255, "31-40%": .355,
                        "41-50%": .455, "51-60%": .555, "61-70%": .655, "71-80%": .755,
                        "81-90%": .855, "91-100%": .955})
                    frame["y"] = frame.y.astype("int8")
                    frame = frame[["record_id", "origin_program", "y", "fold", *FEATURES]]
                    for rid in frame.record_id.astype(int):
                        if selected_seen[rid]:
                            raise ValueError(f"Fold stream repeated record ID {rid}")
                        selected_seen[rid] = True
                    origins_seen.update(frame.origin_program.astype(str).unique())
                    for label, count in frame.y.value_counts().items():
                        class_counts[int(label)] += int(count)
                    total += len(frame)
                    yield frame
                if part_rows != int(part_info["rows"]):
                    raise ValueError(f"QASM chunk row count mismatch: {path}")
            if int(seen_cache.sum()) != self.expected_rows:
                raise ValueError(f"QASM feature cache covers {int(seen_cache.sum())} rows, expected {self.expected_rows}")
            expected = self.expected_rows - int((self._fold_by_id == fold).sum()) if role == "training" else int((self._fold_by_id == fold).sum())
            if total != expected:
                raise ValueError(f"Fold {fold} {role} stream covered {total} rows, expected {expected}")
            self.last_stream_audit = {"fold": fold, "role": role, "records_seen": total,
                "unique_ids": int(selected_seen.sum()), "origins": len(origins_seen),
                "class_counts": {str(y): count for y, count in class_counts.items()},
                "skipped_rows": 0, "feature_errors": 0,
                "feature_fingerprint": self.audit["qasm_manifest_sha256"]}
        finally:
            con.close()
