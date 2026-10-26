"""Shared QMutBench QASM mutation-diff parsing and feature definitions."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from qmutbench_validation import parse_qasm_file


class UnsupportedQASMGateError(ValueError):
    """A valid OpenQASM operation is outside the feature extractor's gate schema."""

PARSER_ARITY = {
    "id": (1, 0), "x": (1, 0), "y": (1, 0), "z": (1, 0), "h": (1, 0),
    "s": (1, 0), "sdg": (1, 0), "t": (1, 0), "tdg": (1, 0), "sx": (1, 0),
    "sxdg": (1, 0), "rx": (1, 1), "ry": (1, 1), "rz": (1, 1), "p": (1, 1),
    "u1": (1, 1), "u2": (1, 2), "u3": (1, 3), "u": (1, 3),
    "cx": (2, 0), "cy": (2, 0), "cz": (2, 0), "ch": (2, 0), "swap": (2, 0),
    "cp": (2, 1), "cu1": (2, 1), "cu2": (2, 2), "cu3": (2, 3), "cu": (2, 4),
    "crx": (2, 1), "cry": (2, 1), "crz": (2, 1), "rxx": (2, 1), "ryy": (2, 1),
    "rzz": (2, 1), "rzx": (2, 1), "ccx": (3, 0), "rccx": (3, 0), "cswap": (3, 0),
}


def circuit_signature_from_qasm_text(source: str) -> tuple[list[tuple[str, tuple[int, ...]]], int]:
    if not isinstance(source, str) or not source.strip():
        raise ValueError("QASM source must be a non-empty string")
    with tempfile.TemporaryDirectory(prefix="qmutbench-qasm-") as directory:
        path = Path(directory) / "record.qasm"
        path.write_text(source, encoding="utf-8")
        parsed, qubit_count = parse_qasm_file(path)
    result = []
    for operation in parsed:
        name = str(operation[0]).lower()
        if name in {"measure", "barrier"}:
            continue
        if name not in PARSER_ARITY:
            raise UnsupportedQASMGateError(
                f"Unsupported OpenQASM gate {name!r}; mutation features support {len(PARSER_ARITY)} gate names."
            )
        arity, parameter_count = PARSER_ARITY[name]
        if len(operation) != 1 + arity + parameter_count:
            raise ValueError(f"Unexpected parsed operation representation for {name!r}")
        result.append((name, tuple(int(q) for q in operation[1:1 + arity])))
    return result, qubit_count


def operations_from_qasm_text(source: str) -> list[tuple[str, tuple[int, ...]]]:
    """Return outcome-independent gate/wire signatures, excluding measure/barrier."""
    return circuit_signature_from_qasm_text(source)[0]


def mutation_diff_features(original: list[tuple[str, tuple[int, ...]]],
                           mutant: list[tuple[str, tuple[int, ...]]],
                           operation: str, expected_gate: str) -> dict[str, Any]:
    """Conservative common-prefix/suffix alignment on gate and wire signatures."""
    prefix = 0
    while prefix < min(len(original), len(mutant)) and original[prefix] == mutant[prefix]:
        prefix += 1
    suffix = 0
    while (suffix < min(len(original) - prefix, len(mutant) - prefix)
           and original[len(original) - 1 - suffix] == mutant[len(mutant) - 1 - suffix]):
        suffix += 1
    original_end = len(original) - suffix if suffix else len(original)
    mutant_end = len(mutant) - suffix if suffix else len(mutant)
    removed, added = original[prefix:original_end], mutant[prefix:mutant_end]
    orig_span, mut_span = len(removed), len(added)
    operation, expected_gate = str(operation), str(expected_gate).lower()
    changed_side = added if operation in {"Add", "Replace"} else removed
    gate_match = any(name == expected_gate for name, _ in changed_side)
    exact_shapes = {"Add": (0, 1), "Remove": (1, 0), "Replace": (1, 1)}
    exact_single = (orig_span, mut_span) == exact_shapes.get(operation, (-1, -1)) and gate_match
    local = orig_span + mut_span <= 3 and gate_match
    if exact_single:
        alignment = {"Add":"single_insert", "Remove":"single_delete", "Replace":"single_replace"}[operation]
    elif local:
        alignment = "near_multi_edit"
    elif orig_span + mut_span == 0:
        alignment = "no_detected_edit"
    else:
        alignment = "complex_or_unaligned"
    if local:
        previous = original[prefix - 1][0] if prefix else "circuit_start"
        following = original[original_end][0] if original_end < len(original) else "circuit_end"
    else:
        previous = following = "unknown_complex"
    removed_gate = removed[0][0] if orig_span == 1 else ("none" if orig_span == 0 else "multi")
    added_gate = added[0][0] if mut_span == 1 else ("none" if mut_span == 0 else "multi")
    qubit_equal = float(set(removed[0][1]) == set(added[0][1])) if orig_span == 1 and mut_span == 1 else -1.0
    touched = {q for _, wires in removed + added for q in wires}
    removed_arity = len(removed[0][1]) if orig_span == 1 else (-1 if orig_span == 0 else -2)
    added_arity = len(added[0][1]) if mut_span == 1 else (-1 if mut_span == 0 else -2)
    return {
        "qasm_diff_orig_span": orig_span, "qasm_diff_mut_span": mut_span,
        "qasm_diff_total_span": orig_span + mut_span,
        "qasm_diff_common_prefix_fraction": prefix / max(1, len(original)),
        "qasm_diff_common_suffix_fraction": suffix / max(1, len(original)),
        "qasm_diff_position_fraction": prefix / max(1, len(original)),
        "qasm_diff_expected_gate_match": int(gate_match),
        "qasm_diff_qubit_set_equal": qubit_equal, "qasm_diff_qubits_touched": len(touched),
        "qasm_diff_removed_arity": removed_arity, "qasm_diff_added_arity": added_arity,
        "qasm_diff_exact_single_edit": int(exact_single), "qasm_diff_is_local": int(local),
        "qasm_diff_removed_gate": removed_gate, "qasm_diff_added_gate": added_gate,
        "qasm_diff_alignment_class": alignment, "qasm_diff_local_prev_gate": previous,
        "qasm_diff_local_next_gate": following,
    }
