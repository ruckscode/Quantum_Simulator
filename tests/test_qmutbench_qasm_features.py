from __future__ import annotations

import pytest

from experiments.qmutbench_qasm_features import FEATURE_COLUMNS, circuit_features
from experiments.qmutbench_qasm_model_comparison import assert_group_fold_isolation
import pandas as pd


def _qasm(tmp_path, name, body):
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def test_native_gate_depth_counts_and_barrier_synchronization(tmp_path):
    path = _qasm(tmp_path, "known.qasm", """OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[3]; creg c[3];
    h q[0]; x q[1]; cx q[0],q[1];
    barrier q[0],q[1],q[2];
    ccx q[0],q[1],q[2];
    measure q -> c;
    """)
    result = circuit_features(path)
    assert result["gate_count"] == 4
    assert result["single_qubit_count"] == 2
    assert result["two_qubit_count"] == 1
    assert result["three_plus_count"] == 1
    assert result["depth"] == 3
    assert result["two_plus_fraction"] == pytest.approx(0.5)


def test_parameterized_gates_exclude_measurements_and_keep_native_ops(tmp_path):
    path = _qasm(tmp_path, "params.qasm", """OPENQASM 2.0;
    qreg q[2]; creg c[2];
    u(pi/2, 0, pi) q[0]; cp(pi/4) q[0],q[1];
    measure q -> c;
    """)
    result = circuit_features(path)
    assert result["gate_count"] == 2
    assert result["depth"] == 2
    assert result["u_fraction"] == pytest.approx(0.5)
    assert result["cp_fraction"] == pytest.approx(0.5)


def test_unsupported_gate_fails_instead_of_fabricating_features(tmp_path):
    path = _qasm(tmp_path, "unsupported.qasm", "OPENQASM 2.0; qreg q[1]; custom q[0];")
    with pytest.raises(ValueError, match="unsupported gate arity"):
        circuit_features(path)


def test_feature_schema_excludes_target_group_and_row_identifier():
    assert not {"Killed", "Origin_program", "record_id", "Unnamed: 0"}.intersection(FEATURE_COLUMNS)
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))


def test_group_fold_validation_rejects_origin_overlap():
    valid = pd.DataFrame({"record_id": [1, 2, 3], "origin_program": ["a", "a", "b"],
                          "fold": [1, 1, 2], "y": [0, 1, 0]})
    assert_group_fold_isolation(valid)
    invalid = valid.copy()
    invalid.loc[1, "fold"] = 2
    with pytest.raises(ValueError, match="Origin_program"):
        assert_group_fold_isolation(invalid)
