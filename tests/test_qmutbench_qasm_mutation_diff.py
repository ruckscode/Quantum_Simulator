from __future__ import annotations

from experiments.qmutbench_qasm_mutation_diff import mutation_diff_features
from qmutbench_ml.qasm_mutation_diff import (UnsupportedQASMGateError,
    circuit_signature_from_qasm_text, operations_from_qasm_text)
import pytest


def test_add_edit_recovers_expected_gate_and_neighbor_context():
    original = [("h", (0,)), ("cx", (0, 1)), ("ry", (1,))]
    mutant = [("h", (0,)), ("x", (0,)), ("cx", (0, 1)), ("ry", (1,))]
    result = mutation_diff_features(original, mutant, "Add", "x")
    assert result["qasm_diff_alignment_class"] == "single_insert"
    assert result["qasm_diff_exact_single_edit"] == 1
    assert result["qasm_diff_expected_gate_match"] == 1
    assert result["qasm_diff_added_gate"] == "x"
    assert result["qasm_diff_local_prev_gate"] == "h"
    assert result["qasm_diff_local_next_gate"] == "cx"


def test_remove_edit_recovers_original_gate():
    original = [("h", (0,)), ("cx", (0, 1)), ("ry", (1,))]
    mutant = [("h", (0,)), ("ry", (1,))]
    result = mutation_diff_features(original, mutant, "Remove", "cx")
    assert result["qasm_diff_alignment_class"] == "single_delete"
    assert result["qasm_diff_removed_gate"] == "cx"
    assert result["qasm_diff_expected_gate_match"] == 1
    assert result["qasm_diff_local_prev_gate"] == "h"
    assert result["qasm_diff_local_next_gate"] == "ry"


def test_replace_edit_recovers_gate_pair_and_wire_relationship():
    original = [("h", (0,)), ("cx", (0, 1)), ("ry", (1,))]
    mutant = [("h", (0,)), ("cz", (0, 1)), ("ry", (1,))]
    result = mutation_diff_features(original, mutant, "Replace", "cz")
    assert result["qasm_diff_alignment_class"] == "single_replace"
    assert result["qasm_diff_removed_gate"] == "cx"
    assert result["qasm_diff_added_gate"] == "cz"
    assert result["qasm_diff_qubit_set_equal"] == 1.0


def test_complex_edit_is_explicitly_marked_and_context_not_claimed():
    original = [("h", (0,)), ("cx", (0, 1)), ("ry", (1,)), ("x", (0,))]
    mutant = [("z", (0,)), ("cx", (0, 1)), ("rz", (1,)), ("y", (0,))]
    result = mutation_diff_features(original, mutant, "Replace", "z")
    assert result["qasm_diff_alignment_class"] == "complex_or_unaligned"
    assert result["qasm_diff_is_local"] == 0
    assert result["qasm_diff_local_prev_gate"] == "unknown_complex"
    assert result["qasm_diff_local_next_gate"] == "unknown_complex"


def test_qasm_text_parser_uses_gate_wire_signatures_and_ignores_measurement():
    source = '''OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    h q[0];
    cx q[0],q[1];
    measure q -> c;
    '''
    assert operations_from_qasm_text(source) == [("h", (0,)), ("cx", (0, 1))]


def test_unsupported_gate_and_conditional_qasm_fail_with_clear_errors():
    unsupported = 'OPENQASM 2.0; qreg q[1]; customgate q[0];'
    with pytest.raises(UnsupportedQASMGateError, match="Unsupported OpenQASM gate 'customgate'"):
        operations_from_qasm_text(unsupported)
    conditional = 'OPENQASM 2.0; qreg q[1]; creg c[1]; if (c==1) x q[0];'
    with pytest.raises(ValueError, match="malformed QASM parameter"):
        operations_from_qasm_text(conditional)


def test_circuit_signature_reports_qubit_count_for_pair_validation():
    source = 'OPENQASM 2.0; qreg q[3]; h q[0];'
    operations, qubits = circuit_signature_from_qasm_text(source)
    assert operations == [("h", (0,))]
    assert qubits == 3
