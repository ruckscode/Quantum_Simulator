"""Read-only adapter for the verified metadata and QASM-diff cache."""
from __future__ import annotations

import hashlib
import json
import gc
import csv
from pathlib import Path

import pandas as pd

TARGET_MAP = {"WOO": 1, "OPO": 1, "Survivor": 0}
METADATA_FIELDS = [
    "algorithm", "operation", "gate", "dominant_state", "qubits", "gates",
    "Position_int", "position",
]
QASM_NUMERIC = [
    "qasm_diff_orig_span", "qasm_diff_mut_span", "qasm_diff_total_span",
    "qasm_diff_expected_gate_match", "qasm_diff_qubit_set_equal",
    "qasm_diff_qubits_touched", "qasm_diff_removed_arity",
    "qasm_diff_added_arity", "qasm_diff_exact_single_edit",
    "qasm_diff_is_local",
]
QASM_CATEGORICAL = [
    "qasm_diff_removed_gate", "qasm_diff_added_gate", "qasm_diff_alignment_class",
    "qasm_diff_local_prev_gate", "qasm_diff_local_next_gate",
]
FEATURES = ["algorithm", "operation", "gate", "dominant_state", "qubits", "gates",
            "Position_int", "position_fraction", *QASM_NUMERIC, *QASM_CATEGORICAL]
CATEGORICAL = ["algorithm", "operation", "gate", "dominant_state", *QASM_CATEGORICAL]
NUMERIC = ["qubits", "gates", "Position_int", "position_fraction", *QASM_NUMERIC]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_development_records(root: Path, origins=None, record_ids=None) -> tuple[pd.DataFrame, dict]:
    """Load aligned development metadata/features and audit label/group integrity."""
    raw_path = root / "data/qmutbench_ingestion/raw/official_all_data_20261015.csv"
    oof_path = root / "data/qmutbench_ml/master_optimization_v1/oof_predictions.csv"
    cache = root / "data/qmutbench_ml/qasm_mutation_diff_v1"
    oof = pd.read_csv(oof_path, usecols=["record_id", "origin_program", "y_true", "fold"])
    oof["record_id"] = oof.record_id.astype("int64")
    if oof.record_id.duplicated().any() or len(oof) != 578117:
        raise ValueError("Saved development OOF rows are duplicated or do not match the verified 578,117-row split")
    development_rows = len(oof)
    development_origins = int(oof.origin_program.nunique())
    if origins is not None:
        origins = set(map(str, origins))
        if not origins or not origins <= set(oof.origin_program.astype(str).unique()):
            raise ValueError("Requested origin filter is empty or outside the verified development manifest")
        oof = oof[oof.origin_program.astype(str).isin(origins)].copy()
    if record_ids is not None:
        record_ids = set(map(int, record_ids))
        if not record_ids or not record_ids <= set(oof.record_id.astype(int)):
            raise ValueError("Requested record IDs are empty or outside the selected development origins")
        oof = oof[oof.record_id.isin(record_ids)].copy()
    development_ids = set(oof.record_id.astype(int))
    # Inspect only the row identifier before filtering. Target, group, and
    # feature columns are accessed only for IDs in the saved development OOF.
    source_rows = []
    with raw_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        id_i = header.index("") if header and header[0] == "" else header.index("Unnamed: 0")
        wanted = {name: header.index(name) for name in ["Origin_program", "Killed", *METADATA_FIELDS]}
        for row in reader:
            record_id = int(row[id_i])
            if record_id in development_ids:
                source_rows.append({"record_id": record_id,
                    **{name: row[index] for name, index in wanted.items()}})
    raw = pd.DataFrame(source_rows).rename(columns={"Origin_program": "origin_program"})
    raw = raw.replace("", float("nan"))
    raw["origin_program"] = raw.origin_program.astype("category")
    for name in ("qubits", "gates", "Position_int"):
        raw[name] = pd.to_numeric(raw[name], errors="coerce")
    raw["Killed"] = raw.Killed.astype("category")
    raw[["algorithm", "operation", "gate", "dominant_state"]] = raw[["algorithm", "operation", "gate", "dominant_state"]].astype("category")
    if len(raw) != len(oof):
        raise ValueError("Raw metadata does not cover the saved development IDs exactly")
    observed = set(raw.Killed.dropna().unique())
    if not observed <= set(TARGET_MAP):
        raise ValueError(f"Unexpected target labels: {sorted(observed)}")
    raw["y_from_source"] = raw.Killed.map(TARGET_MAP).astype("int8")
    aligned = raw.merge(oof, on=["record_id", "origin_program"], validate="one_to_one")
    del raw, oof
    gc.collect()
    if not (aligned.y_from_source == aligned.y_true).all():
        raise ValueError("Source label encoding disagrees with saved development OOF labels")

    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    expected_source = "3f4547868673c583f047cee19d27c7bc97c64f15bb0586098ada778a3a0ff778"
    actual_source = sha256(raw_path)
    if actual_source != expected_source or manifest.get("source_sha256") != expected_source:
        raise ValueError("Metadata or QASM cache does not match the verified source dataset checksum")
    parts = []
    needed = set(aligned.record_id.astype(int))
    cache_columns = ["record_id", *QASM_NUMERIC, *QASM_CATEGORICAL]
    cache_dtypes = {**{name: "float32" for name in QASM_NUMERIC},
                    **{name: "category" for name in QASM_CATEGORICAL}}
    for chunk in sorted(cache.glob("diff_*.csv.gz")):
        for part in pd.read_csv(chunk, usecols=cache_columns, dtype=cache_dtypes, chunksize=50000):
            selected = part[part.record_id.isin(needed)]
            if not selected.empty:
                parts.append(selected)
    qasm = pd.concat(parts, ignore_index=True)
    if len(qasm) != len(aligned) or qasm.record_id.duplicated().any():
        raise ValueError("QASM-diff cache does not align one-to-one with development metadata")
    qasm[QASM_CATEGORICAL] = qasm[QASM_CATEGORICAL].astype("category")
    frame = aligned.merge(qasm, on="record_id", validate="one_to_one")
    del aligned, qasm, parts
    gc.collect()
    frame["position_fraction"] = frame.position.map({
        "0-10%": .05, "11-20%": .155, "21-30%": .255, "31-40%": .355,
        "41-50%": .455, "51-60%": .555, "61-70%": .655, "71-80%": .755,
        "81-90%": .855, "91-100%": .955,
    })
    if frame["position_fraction"].isna().any():
        raise ValueError("Unknown source position band")
    frame["y"] = frame.y_from_source.astype("int8")
    if len(FEATURES) != 23:
        raise AssertionError("Feature parity contract should contain the model's 23 inputs")
    audit = {
        "source_path": str(raw_path.relative_to(root)),
        "source_sha256": actual_source,
        "development_rows": int(development_rows),
        "development_origins": development_origins,
        "loaded_development_rows": int(len(frame)),
        "loaded_development_origins": int(frame.origin_program.nunique()),
        "qasm_cache_manifest_sha256": sha256(cache / "manifest.json"),
        "qasm_cache_rows": int(manifest.get("feature_rows", -1)),
        "raw_source_scanned_sequentially": True,
        "target_and_features_read_only_after_development_id_match": True,
        "frozen_test_rows_retained": 0,
        "feature_schema_23": FEATURES,
        "target_map": TARGET_MAP,
        "label_counts": {str(k): int(v) for k, v in frame.y.value_counts().sort_index().items()},
        "saved_fold_counts": {str(k): int(v) for k, v in frame.fold.value_counts().sort_index().items()},
    }
    return frame, audit
