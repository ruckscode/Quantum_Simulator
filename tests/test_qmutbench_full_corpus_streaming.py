from __future__ import annotations

import json

import pandas as pd
import pytest

from experiments import qmutbench_full_corpus_streaming as stream


def _qasm(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def test_safe_relative_paths_matches_qmutbench_naming():
    original, mutant = stream.safe_relative_paths({
        "algorithm": "ghz", "qubits": 2, "operation": "Add",
        "gate": "h", "Position_int": 1,
    })
    assert original == "Origin_programs/ghz_2_qubits.qasm"
    assert mutant == "Mutated_programs/Mutants_ghz_2_qubits/AddGate_h_inGap_1_.qasm"


def test_safe_relative_paths_rejects_unsafe_tokens():
    with pytest.raises(ValueError, match="Unsafe"):
        stream.safe_relative_paths({
            "algorithm": "../escape", "qubits": 2, "operation": "Add",
            "gate": "h", "Position_int": 1,
        })


def test_ram_floor_stops_before_work(monkeypatch):
    monkeypatch.setattr(stream, "available_ram_bytes", lambda: 100 * 1024**2)
    with pytest.raises(MemoryError, match="safety floor"):
        stream.require_ram_floor(1024)


def test_feature_rows_do_not_include_target_or_group_as_predictors():
    result = stream._feature_row(
        4, "origin-a", "origin.qasm", "mutant.qasm",
        {"gate_count": 2, "depth": 2, "two_plus_fraction": 0.5},
        {"gate_count": 3, "depth": 3, "two_plus_fraction": 1.0},
        3,
    )
    assert result["record_id"] == 4
    assert result["origin_program"] == "origin-a"  # Grouping only.
    assert result["source_gate_count_match"] == 1
    assert "Killed" not in result
    assert "label" not in result
    assert result["qasm_delta_gate_count"] == 1


def test_one_chunk_extracts_features_and_resumes_without_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(stream, "available_ram_bytes", lambda: 4 * 1024**3)
    metadata = tmp_path / "metadata.csv"
    qasm_root = tmp_path / "data"
    output = tmp_path / "features"

    _qasm(qasm_root / "Origin_programs/ghz_2_qubits.qasm", '''OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0],q[1];
''')
    _qasm(qasm_root / "Mutated_programs/Mutants_ghz_2_qubits/AddGate_x_inGap_1_.qasm", '''OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
x q[1];
cx q[0],q[1];
''')
    pd.DataFrame([{
        "Unnamed: 0": 0, "Origin_program": "ghz_2_qubits",
        "algorithm": "ghz", "qubits": 2, "gates": 3,
        "operation": "Add", "gate": "x", "Position_int": 1,
        "Killed": "WOO",
    }]).to_csv(metadata, index=False)

    result = stream.extract(metadata, qasm_root, output, chunk_rows=1,
                            min_available_mib=1024, max_chunks=1)
    assert result["status"] == "partial"
    assert result["records_requested"] == 1
    assert result["feature_rows"] == 1
    assert result["parse_error_rows"] == 0

    part = pd.read_csv(output / "features_00000.csv.gz")
    assert len(part) == 1
    assert part.loc[0, "record_id"] == 0
    assert part.loc[0, "qasm_mut_gate_count"] == 3
    assert "Killed" not in part.columns
    assert "label" not in part.columns

    manifest_before = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_before["chunks"][0]["sha256"]

    # Same inputs and configuration: completed chunk is reused, not overwritten.
    result2 = stream.extract(metadata, qasm_root, output, chunk_rows=1,
                             min_available_mib=1024, max_chunks=1)
    assert result2["feature_rows"] == 1
    assert result2["parse_error_rows"] == 0
