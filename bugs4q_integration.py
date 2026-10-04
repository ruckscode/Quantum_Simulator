"""Source-backed integration of selected Bugs4Q Qiskit examples.

The adapter reads the official Python sources without importing/executing them,
extracts only the circuit-building calls, and sends translated operations through
the project's simulator, statistics, diagnosis, and localization APIs.
"""
from __future__ import annotations

import ast
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from diagnosis_engine import diagnose_execution
from quantum_simulator import QuantumSimulator
from statistics import compare_distributions


PROJECT_ROOT = Path(__file__).resolve().parent
OFFICIAL_ROOT = PROJECT_ROOT / "data/Bugs4Q_official/extracted/Zenodo-Framework/Bugs4Q-Framework/qiskit"
INTEGRATED_CASES = (8, 12, 17, 25, 26, 30, 31, 39)
_SIMPLE_ARITY = {"h": 1, "x": 1, "z": 1, "y": 1, "s": 1, "t": 1, "id": 1,
                 "cx": 2, "cnot": 2, "cz": 2, "swap": 2, "ccx": 3,
                 "reset": 1, "barrier": -1, "crz": 3}


def _safe_eval(node: ast.AST, names: dict[str, Any] | None = None) -> Any:
    names = {"pi": math.pi, **(names or {})}
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in names:
        return names[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _safe_eval(node.operand, names)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp):
        left, right = _safe_eval(node.left, names), _safe_eval(node.right, names)
        if isinstance(node.op, ast.Add): return left + right
        if isinstance(node.op, ast.Sub): return left - right
        if isinstance(node.op, ast.Mult): return left * right
        if isinstance(node.op, ast.Div): return left / right
        if isinstance(node.op, ast.Pow): return left ** right
    raise ValueError(f"unsupported source expression: {ast.dump(node, include_attributes=False)}")


def _calls(node: ast.AST, env: dict[str, Any]) -> Iterable[tuple[ast.Call, dict[str, Any]]]:
    """Yield circuit calls in source order, expanding literal range loops."""
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda):
        return
    if isinstance(node, ast.For) and isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range":
        values = [_safe_eval(arg, env) for arg in node.iter.args]
        sequence = range(*values)
        for value in sequence:
            next_env = dict(env)
            if isinstance(node.target, ast.Name): next_env[node.target.id] = value
            for child in node.body:
                yield from _calls(child, next_env)
        for child in node.orelse:
            yield from _calls(child, env)
        return
    if isinstance(node, ast.Call):
        yield node, dict(env)
        # Calls nested as arguments are not separate circuit operations.
        return
    for child in ast.iter_child_nodes(node):
        yield from _calls(child, env)


def _register_layout(tree: ast.Module, case_number: int) -> tuple[dict[str, tuple[int, ...]], int]:
    sizes: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name) or value.func.id != "QuantumRegister":
                continue
            target = next((item for item in targets if isinstance(item, ast.Name)), None)
            if target is not None:
                sizes[target.id] = int(_safe_eval(value.args[0]))
    if case_number == 30:
        # QuantumCircuit is constructed in order: search(4), classical(4), m(1), ancilla(3).
        names = ("search_register", "m_qubit", "ancillaries")
        layout, index = {}, 0
        for name in names:
            size = sizes[name]
            layout[name] = tuple(range(index, index + size))
            index += size
        return layout, index
    return {}, 0


def _resolve_qubit(node: ast.AST, env: dict[str, Any], layout: dict[str, tuple[int, ...]]) -> int:
    if isinstance(node, ast.Name):
        value = env.get(node.id)
        if isinstance(value, int): return value
        if node.id in layout and len(layout[node.id]) == 1: return layout[node.id][0]
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        name = node.value.id
        if name == "qc" and isinstance(node.slice, ast.Constant):
            return int(_safe_eval(node.slice, env))
        if name in layout:
            return layout[name][int(_safe_eval(node.slice, env))]
        if isinstance(node.slice, ast.Constant):
            return int(_safe_eval(node.slice, env))
    return int(_safe_eval(node, env))


def _resolve_register(node: ast.AST, layout: dict[str, tuple[int, ...]], env: dict[str, Any]) -> list[int]:
    if isinstance(node, ast.Name) and node.id in layout:
        return list(layout[node.id])
    if isinstance(node, ast.Subscript):
        return [_resolve_qubit(node, env, layout)]
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_resolve_qubit(item, env, layout) for item in node.elts]
    raise ValueError(f"unsupported register expression {ast.dump(node, include_attributes=False)}")


def _extract_source(source: str, case_number: int) -> tuple[list[tuple[Any, ...]], int]:
    tree = ast.parse(source)
    if case_number in (12, 17, 31):
        # Preserve Qiskit's explicit qubit-to-classical-bit mapping. Count
        # strings are emitted in descending classical-bit order below.
        operations: list[tuple[Any, ...]] = []
        for node, env in _calls(tree, {"pi": math.pi}):
            if not isinstance(node.func, ast.Attribute):
                continue
            method = node.func.attr.lower()
            owner = node.func.value.id if isinstance(node.func.value, ast.Name) else None
            expected_owner = "circuit" if case_number == 31 else "qc"
            if owner != expected_owner:
                continue
            if method == "x":
                operations.append(("x", _resolve_qubit(node.args[0], env, {})))
            elif method in ("cx", "cnot"):
                operations.append(("cx", _resolve_qubit(node.args[0], env, {}), _resolve_qubit(node.args[1], env, {})))
            elif method == "h":
                arg = node.args[0]
                if isinstance(arg, ast.Name):
                    operations.extend(("h", q) for q in range(2))
                else:
                    operations.append(("h", _resolve_qubit(arg, env, {})))
            elif method == "measure_all":
                operations.append(("measure", [0, 1], [0, 1]))
            elif method == "measure":
                if isinstance(node.args[0], ast.Attribute) and node.args[0].attr == "qubits":
                    qubits = [0, 1]
                    classical = [0, 1]
                else:
                    qubits = [_resolve_qubit(q, env, {}) for q in node.args[0].elts]
                    classical = [_resolve_qubit(q, env, {}) for q in node.args[1].elts]
                operations.append(("measure", qubits, classical))
        return operations, 2 if case_number == 12 else 3
    layout, num_qubits = _register_layout(tree, case_number)
    num_qubits = num_qubits or {8: 4, 25: 3, 26: 2, 39: 4}.get(case_number, 0)
    operations: list[tuple[Any, ...]] = []
    names = {"pi": math.pi}
    for node, call_env in _calls(tree, names):
        if not isinstance(node.func, ast.Attribute):
            continue
        method = node.func.attr.lower()
        owner = node.func.value.id if isinstance(node.func.value, ast.Name) else None
        if case_number == 8:
            if owner != "circ" or method != "crz": continue
            theta = float(_safe_eval(node.args[0], call_env))
            control = _resolve_qubit(node.args[1], call_env, {})
            target = _resolve_qubit(node.args[2], call_env, {})
            # The source builds a 3-qubit Operator and appends it on qc qubits [0,2,3].
            mapped = {0: 0, 1: 2, 2: 3}
            operations.append(("crz", mapped[control], mapped[target], theta))
            num_qubits = 4
            continue
        expected_owner = "circuit" if case_number in (25, 26, 30) else "qc"
        if owner != expected_owner: continue
        if case_number == 25 and method in {"reset", "x", "ccx", "barrier", "measure"}:
            pass
        elif case_number == 26 and method in {"h", "x", "cx", "measure_all"}:
            pass
        elif case_number == 30 and method in {"h", "x", "ccx", "reset", "measure"}:
            pass
        elif case_number == 39 and method in {"h", "cx", "ccx", "measure"}:
            pass
        else:
            continue
        if method == "measure_all":
            measured = list(range(num_qubits))
            operations.append(("measure", measured, list(measured)))
            continue
        args = node.args
        env = call_env
        if method in {"measure", "reset", "barrier", "h", "x", "cx", "ccx"}:
            qubits: list[int]
            if method == "measure":
                qubits = []
            elif case_number == 30 and method in {"h", "x", "reset", "measure"} and args and isinstance(args[0], ast.Name) and args[0].id in layout:
                qubits = list(layout[args[0].id])
            elif case_number == 39 and method == "h" and args and isinstance(args[0], ast.Name) and args[0].id == "qc":
                qubits = list(range(num_qubits))
            elif case_number == 39 and method == "h" and args:
                qubits = [_resolve_qubit(args[0], env, {})]
            else:
                qubits = [_resolve_qubit(arg, env, layout) for arg in args[:_SIMPLE_ARITY[method]]]
            if method == "measure":
                if case_number == 30:
                    # Search register is the only measured quantum register in this case.
                    qubits = list(layout["search_register"])
                elif case_number == 25:
                    qubits = [0, 1, 2]
                elif case_number == 39:
                    qubits = [_resolve_qubit(args[0], env, layout)]
                # The selected cases use identity classical destinations unless
                # an explicit mapping was extracted above. Keep both operands
                # explicit for diagnosis/localization normalization.
                operations.append(("measure", qubits, list(qubits)))
            elif method == "barrier":
                operations.append(("barrier", *qubits))
            elif method == "reset":
                operations.extend(("reset", qubit) for qubit in qubits)
            else:
                if method in {"h", "x"} and len(qubits) > 1:
                    operations.extend((method, qubit) for qubit in qubits)
                else:
                    operations.append((method, *qubits))
    if case_number != 8 and num_qubits == 0:
        if case_number in (25,): num_qubits = 3
        elif case_number == 26: num_qubits = 2
        elif case_number == 39: num_qubits = 4
    return operations, num_qubits


def _apply_operations(operations: list[tuple[Any, ...]], num_qubits: int, shots: int, seed: int) -> dict[str, int]:
    """Execute once on QuantumSimulator, then sample its measured marginal."""
    import numpy as np
    sim = QuantumSimulator(num_qubits, seed=seed)
    selected: list[int] = []
    classical: list[int] = []
    for operation in operations:
        name, *args = operation
        if name == "measure":
            selected.extend(args[0])
            classical.extend(args[1] if len(args) > 1 else args[0])
        elif name == "crz":
            control, target, theta = args
            sim.apply_gate("crz", [int(control), int(target)], float(theta))
        else:
            sim.run_circuit([operation])
    measured = selected if selected else list(range(num_qubits))
    classical = classical if classical else measured
    marginal: dict[str, float] = {}
    for basis, probability in enumerate(sim.get_probabilities()):
        bits = f"{basis:0{num_qubits}b}"
        classical_bits = ["0"] * (max(classical, default=-1) + 1)
        for qubit, clbit in zip(measured, classical):
            # QuantumSimulator's register convention places qubit 0 at the
            # left of the basis string; reverse only the classical display.
            classical_bits[clbit] = bits[qubit]
        outcome = "".join(reversed(classical_bits))
        marginal[outcome] = marginal.get(outcome, 0.0) + float(probability)
    outcomes = sorted(marginal)
    probs = np.asarray([marginal[key] for key in outcomes], dtype=float)
    probs /= probs.sum()
    draws = np.random.default_rng(seed).choice(len(outcomes), size=shots, p=probs)
    counts: dict[str, int] = {}
    for draw in draws:
        key = outcomes[int(draw)]
        counts[key] = counts.get(key, 0) + 1
    return counts


def _distribution(counts: dict[str, int], shots: int) -> dict[str, float]:
    return {key: value / shots for key, value in sorted(counts.items())}


def _diagnosis_view(operations: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    """Expand controlled-RZ for diagnosis/localization's generic runner."""
    result: list[tuple[Any, ...]] = []
    for operation in operations:
        if operation[0] == "crz":
            _, control, target, theta = operation
            # Exact CRZ decomposition into gates accepted by run_circuit.
            result.extend([
                ("cx", control, target),
                ("rz", target, -theta / 2),
                ("cx", control, target),
                ("rz", target, theta / 2),
            ])
        else:
            result.append(operation)
    return result


def _jsonable(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict): return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_jsonable(v) for v in value]
    return value


@dataclass
class Bugs4QIntegrationReport:
    benchmark: str
    cases: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


def _case_metadata(case_number: int, case_dir: Path) -> dict[str, Any]:
    info_path = case_dir / f"info_{case_number}.csv"
    with info_path.open(encoding="utf-8-sig", newline="") as stream:
        row = next(csv.DictReader(stream, delimiter=";"))
    root_relative = case_dir.relative_to(PROJECT_ROOT).as_posix()
    tests = sorted(path.name for path in case_dir.glob(f"test_*{case_number}*.py"))
    return {
        "case_number": case_number,
        "bug_type": row.get("Bug-Type"),
        "source_url": row.get("Url"),
        "buggy_source": f"{root_relative}/buggy_{case_number}.py",
        "fixed_source": f"{root_relative}/fixed_{case_number}.py",
        "modify_file": f"{root_relative}/modify_{case_number}.txt",
        "test_references": [f"{root_relative}/{name}" for name in tests],
    }


def discover_bugs4q_cases(root: str | Path = OFFICIAL_ROOT) -> list[int]:
    base = Path(root)
    return [n for n in INTEGRATED_CASES if all((base / str(n) / filename.format(n=n)).is_file() for filename in ("buggy_{n}.py", "fixed_{n}.py", "info_{n}.csv", "modify_{n}.txt"))]


def run_bugs4q_integration(
    root: str | Path = OFFICIAL_ROOT, *, shots: int = 1024, seed: int = 1729, threshold: float = 0.05,
) -> Bugs4QIntegrationReport:
    base = Path(root)
    records = []
    for number in discover_bugs4q_cases(base):
        case_dir = base / str(number)
        buggy_source = (case_dir / f"buggy_{number}.py").read_text(encoding="utf-8-sig")
        fixed_source = (case_dir / f"fixed_{number}.py").read_text(encoding="utf-8-sig")
        buggy, buggy_n = _extract_source(buggy_source, number)
        fixed, fixed_n = _extract_source(fixed_source, number)
        n = max(buggy_n, fixed_n)
        expected_counts = _apply_operations(fixed, n, shots, seed)
        observed_counts = _apply_operations(buggy, n, shots, seed + 1)
        all_outcomes = sorted(set(expected_counts) | set(observed_counts))
        expected_dist = {key: expected_counts.get(key, 0) / shots for key in all_outcomes}
        observed_dist = {key: observed_counts.get(key, 0) / shots for key in all_outcomes}
        comparison = compare_distributions(expected_dist, observed_dist, threshold=threshold, shots=shots)
        diagnosis = diagnose_execution(
            expected_distribution=expected_dist,
            observed_distribution=observed_dist,
            expected_circuit=_diagnosis_view(fixed),
            observed_circuit=_diagnosis_view(buggy),
            threshold=threshold,
            shots=shots,
        )
        metadata = _case_metadata(number, case_dir)
        metadata["source_level_mutation"] = (case_dir / f"modify_{number}.txt").read_text(encoding="utf-8-sig")
        expected_measurements = [
            {"measured_qubits": list(op[1]), "classical_destinations": list(op[2] if len(op) > 2 else op[1])}
            for op in fixed if op[0] == "measure"
        ]
        observed_measurements = [
            {"measured_qubits": list(op[1]), "classical_destinations": list(op[2] if len(op) > 2 else op[1])}
            for op in buggy if op[0] == "measure"
        ]
        # Measurement operations retain their qubit-to-classical-bit mapping
        # through diagnosis and the current classical-bit-aware localizer.
        localization_compatible = True
        limitations = [
            "Only source operations needed for the selected circuit behavior were extracted; Python/Qiskit wrapper behavior was not modeled.",
            "No benchmark accuracy or diagnosis agreement metric is calculated.",
            "Reported measurement strings use simulator qubit-index ordering; Qiskit classical count-string display ordering may differ.",
        ]
        if number == 8:
            limitations.extend([
                "The source-level CRZ mutation is represented through an internal supported expansion; any internal CX candidate is an implementation-level localization result and must not be described as the original source-level CRZ operation.",
                "For this source input, the CRZ controls remain in |0>, so case 8's observed output distributions are equal even though the operation operands differ.",
                "CRZ is retained in the report; execution uses QuantumSimulator.apply_gate matrix support because run_circuit does not dispatch CRZ reliably.",
            ])
        if number == 30:
            limitations.extend([
                "The X(2) candidate is one genuine component of a broader multi-operation mutation and must not be described as the complete mutation.",
                "Reset uses the existing simulator's ideal reset implementation.",
            ])
        records.append({
            "case_number": number,
            "case_id": f"Bugs4Q-{number}",
            "execution_status": "COMPLETED",
            "bugs4q_metadata": metadata,
            "reference_fixed_circuit": fixed,
            "buggy_circuit": buggy,
            "expected_distribution": expected_dist,
            "observed_distribution": observed_dist,
            "behavioral_comparison": _jsonable(comparison),
            "system_diagnosis": {
                "category": diagnosis.category.value,
                "summary": diagnosis.summary,
                "anomaly_detected": diagnosis.anomaly_detected,
                "evidence": diagnosis.evidence,
            },
            "program_localization_candidates": [_jsonable(candidate) for candidate in diagnosis.program_evidence_candidates],
            "localization_evidence": {
                "status": "AVAILABLE" if localization_compatible else "UNAVAILABLE",
                "note": (
                    "The current localizer compares measured qubits and classical-bit destinations."
                    if not localization_compatible else
                    "The current localizer compares translated gate operations and preserves measured qubits plus expected and observed classical-bit destinations."
                ),
                "candidates": [_jsonable(candidate) for candidate in diagnosis.program_evidence_candidates],
                "expected_measurements": expected_measurements,
                "observed_measurements": observed_measurements,
            },
            "ground_truth": {
                "available": False,
                "status": "MISSING",
                "source": None,
                "note": "Bugs4Q bug-type metadata is preserved separately and is not system-diagnosis ground truth.",
            },
            "limitations": limitations,
        })
    return Bugs4QIntegrationReport("Bugs4Q", records)


def write_bugs4q_integration_report(path: str | Path, **kwargs: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(run_bugs4q_integration(**kwargs).to_dict(), indent=2), encoding="utf-8")
    return destination
