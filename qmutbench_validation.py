"""Independent QMutBench dataset adapter for the existing diagnosis pipeline.

The adapter requires explicit QMutBench provenance plus case-like content. A
filename containing only "mutation" is never sufficient. Mutation metadata and
benchmark labels are retained separately and are not passed to diagnosis.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import operator
import re
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from benchmark_validation import BenchmarkAdapter, NormalizedBenchmarkCase
from diagnosis_engine import DiagnosisCategory, DiagnosisResult, diagnose_execution
from gates import get_gate
from quantum_simulator import QuantumSimulator
from statistics import compare_distributions


CASE_SUFFIXES = {".json", ".jsonl", ".ndjson", ".csv"}
TEXT_SUFFIXES = CASE_SUFFIXES | {".qasm", ".txt", ".yaml", ".yml"}
EXCLUDED_PARTS = {".venv", ".git", "__pycache__", ".pytest_cache", ".pytest-tmp", "tests", "reports"}
SINGLE_GATES = {"i", "id", "identity", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx", "sxdg"}
ROTATION_GATES = {"rx", "ry", "rz", "p", "u1"}
TWO_GATES = {"cx", "cnot", "cz", "swap", "ch"}
TWO_QUBIT_ROTATIONS = {"rxx"}
THREE_GATES = {"ccx", "toffoli", "cswap", "fredkin"}
PARAMETERIZED_ARITY = {"rx": 1, "ry": 1, "rz": 1, "p": 1, "u1": 1, "u2": 2, "u3": 3}
CONTROLLED_PARAMETER_GATES = {"cp", "crx", "cry", "crz"}


class QMutBenchParseError(ValueError):
    pass


class QMutBenchUnsupportedError(NotImplementedError):
    pass


@dataclass
class QMutBenchCase:
    case_id: str
    raw: Mapping[str, Any]
    source_files: list[str]
    parse_error: str | None = None


@dataclass
class QMutBenchDataset:
    root: str
    data_files: list[str] = field(default_factory=list)
    metadata_files: list[str] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)
    cases: list[QMutBenchCase] = field(default_factory=list)
    issues: list[dict[str, str]] = field(default_factory=list)
    provenance_verified: bool = False
    test_fixture_data: bool = False
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class QMutBenchCaseResult:
    benchmark: str
    case_id: str
    source_files: list[str]
    support_status: str
    supported: bool
    execution_status: str
    ground_truth_status: str
    benchmark_ground_truth: dict[str, Any]
    system_diagnosis: dict[str, Any] | None
    diagnosis_ground_truth_compatible: bool
    case_level_category_match: bool | None
    reference_circuit: Any
    mutant_circuit: Any
    mutation_metadata: dict[str, Any] | None
    expected_information: Any
    source_level_information: dict[str, Any]
    program_candidates: list[dict[str, Any]]
    hardware_candidates: list[dict[str, Any]]
    evidence: list[str]
    errors: list[str] = field(default_factory=list)
    original_program: str | None = None
    mutant_filename: str | None = None
    original_output: dict[str, Any] | None = None
    mutant_output: dict[str, Any] | None = None
    behavioral_comparison: dict[str, Any] | None = None
    behaviorally_different: bool | None = None
    independent_ground_truth_available: bool = False

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass
class QMutBenchValidationReport:
    root: str
    dataset_found: bool
    status: str
    data_files: list[str]
    metadata_files: list[str]
    source_files: list[str]
    provenance_verified: bool
    cases: list[QMutBenchCaseResult]
    issues: list[dict[str, str]]
    test_fixture_data: bool = False

    def to_dict(self) -> dict[str, Any]:
        supported = sum(case.supported for case in self.cases)
        unsupported = sum(case.support_status == "UNSUPPORTED" for case in self.cases)
        parsing_failures = sum(case.support_status == "PARSING_FAILURE" for case in self.cases) + sum(issue.get("status") == "PARSING_FAILURE" for issue in self.issues)
        uncertain = sum(case.ground_truth_status == "UNCERTAIN" for case in self.cases)
        conflicting = sum(case.ground_truth_status == "CONFLICTING" for case in self.cases)
        missing = sum(case.ground_truth_status == "MISSING" for case in self.cases)
        reliable = sum(case.ground_truth_status == "RELIABLE" for case in self.cases)
        compatible = [case for case in self.cases if case.case_level_category_match is not None]
        metrics: dict[str, Any] = {
            "accuracy": None,
            "confusion_matrix": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "eligible_cases": len(compatible),
            "reason": "No reliable, explicitly diagnosis-category-compatible QMutBench ground truth is available.",
        }
        if compatible:
            matches = sum(case.case_level_category_match is True for case in compatible)
            metrics["accuracy"] = matches / len(compatible)
            labels = sorted({str(case.benchmark_ground_truth.get("diagnosis_category")) for case in compatible})
            matrix = {expected: {observed: 0 for observed in labels} for expected in labels}
            for case in compatible:
                expected = str(case.benchmark_ground_truth["diagnosis_category"])
                observed = str(case.system_diagnosis["category"])
                matrix[expected][observed] = matrix[expected].get(observed, 0) + 1
            metrics["confusion_matrix"] = matrix
            metrics["reason"] = "Case-level category metrics use only explicitly reliable and compatible reference labels."

        return {
            "benchmark": "QMutBench",
            "status": self.status,
            "dataset_type": "TEST_FIXTURE" if self.test_fixture_data else ("REAL_DATA" if self.dataset_found else "UNAVAILABLE"),
            "dataset_root": self.root,
            "dataset_found": self.dataset_found,
            "provenance_verified": self.provenance_verified,
            "files_discovered": {
                "data": list(self.data_files),
                "metadata": list(self.metadata_files),
                "source_reference": list(self.source_files),
            },
            "total_cases": len(self.cases),
            "supported_cases": supported,
            "unsupported_cases": unsupported,
            "parsing_failures": parsing_failures,
            "uncertain_cases": uncertain,
            "conflicting_cases": conflicting,
            "missing_ground_truth_cases": missing,
            "reliable_ground_truth_cases": reliable,
            "executed_cases": sum(case.execution_status == "COMPLETED" for case in self.cases),
            "behaviorally_different_cases": sum(case.behaviorally_different is True for case in self.cases),
            "behaviorally_equivalent_cases": sum(case.behaviorally_different is False for case in self.cases),
            "diagnosed_cases": sum(case.system_diagnosis is not None for case in self.cases),
            "independent_ground_truth_available": any(case.independent_ground_truth_available for case in self.cases),
            "metrics": metrics,
            "cases": [case.to_dict() for case in self.cases],
            "issues": list(self.issues),
            "notes": [
                "Mutation metadata is preserved separately and is not supplied to the diagnosis engine as a label.",
                "Benchmark ground truth and system diagnosis are reported side by side; mutation outcomes are not mapped to diagnosis categories.",
                "Test fixtures are not QMutBench results.",
            ],
        }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _normalized_key(value: Any) -> str:
    return "".join(character.lower() for character in str(value) if character.isalnum())


def _contains_qmutbench_metadata(value: Any) -> bool:
    provenance_keys = {"benchmark", "benchmarkname", "dataset", "datasetname", "datasetid", "provenance", "sourcebenchmark"}
    if isinstance(value, Mapping):
        for key, child in value.items():
            if _normalized_key(key) in provenance_keys and isinstance(child, str) and "qmutbench" in _normalized_key(child):
                return True
            if _contains_qmutbench_metadata(child):
                return True
    elif isinstance(value, list):
        return any(_contains_qmutbench_metadata(item) for item in value)
    return False


def _case_like(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    keys = {_normalized_key(key) for key in value}
    program = value.get("program")
    has_original = bool(keys & {"original", "reference", "originalcircuit", "referencecircuit", "sourcecircuit", "originalprogram", "referenceprogram", "correctcircuit"})
    has_mutant = bool(keys & {"mutantcircuit", "mutatedcircuit", "mutantprogram", "mutatedprogram", "mutant"})
    if isinstance(program, Mapping):
        program_keys = {_normalized_key(key) for key in program}
        has_original = has_original or bool(program_keys & {"original", "reference", "source", "correct"})
        has_mutant = has_mutant or bool(program_keys & {"mutant", "mutated", "modified"})
    return has_original or has_mutant or (bool(keys & {"caseid", "id", "benchmarkcaseid"}) and bool(keys & {"mutation", "mutationoperator", "original", "mutant"}))


def _collection(document: Any) -> tuple[list[Mapping[str, Any]], bool]:
    if isinstance(document, list):
        return [item for item in document if isinstance(item, Mapping)], bool(document)
    if not isinstance(document, Mapping):
        return [], False
    for key in ("cases", "instances", "benchmarks", "qmutbench_cases", "mutants"):
        if key in document:
            collection = document[key]
            if isinstance(collection, Mapping):
                rows = []
                for case_id, value in collection.items():
                    if isinstance(value, Mapping):
                        rows.append({"case_id": str(case_id), **value})
                return rows, True
            if isinstance(collection, list):
                return [item for item in collection if isinstance(item, Mapping)], True
            return [], True
    if _case_like(document):
        return [document], True
    mapping_rows = [value for value in document.values() if isinstance(value, Mapping) and _case_like(value)]
    if mapping_rows:
        return mapping_rows, True
    return [], False


def _decode_cell(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _read_structured_file(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix in {".jsonl", ".ndjson"}:
        rows = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if line.strip():
                item = json.loads(line)
                if not isinstance(item, Mapping):
                    raise ValueError(f"line {line_number} must contain a JSON object")
                rows.append(item)
        return rows
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            return [{key: _decode_cell(value) for key, value in row.items() if key is not None} for row in csv.DictReader(stream)]
    raise ValueError(f"unsupported structured file format: {suffix}")


def _qasm_expression(value: str, case_id: str) -> float:
    """Evaluate the small numeric expression subset used by QMutBench QASM."""
    try:
        tree = ast.parse(value.replace("^", "**"), mode="eval")
        allowed_binary = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow}
        allowed_unary = {ast.UAdd: operator.pos, ast.USub: operator.neg}

        def evaluate(node: ast.AST) -> float:
            if isinstance(node, ast.Expression):
                return evaluate(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return float(node.value)
            if isinstance(node, ast.Name) and node.id == "pi":
                return math.pi
            if isinstance(node, ast.BinOp) and type(node.op) in allowed_binary:
                return allowed_binary[type(node.op)](evaluate(node.left), evaluate(node.right))
            if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_unary:
                return allowed_unary[type(node.op)](evaluate(node.operand))
            raise ValueError("unsupported parameter expression")

        return evaluate(tree)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError) as exc:
        raise QMutBenchParseError(f"{case_id}: malformed QASM parameter {value!r}") from exc


def parse_qasm_file(path: str | Path) -> tuple[list[tuple[Any, ...]], int]:
    """Parse the OpenQASM 2.0 subset used by the checked-in QMutBench programs."""
    path = Path(path)
    case_id = path.name
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise QMutBenchParseError(f"{case_id}: cannot read QASM: {exc}") from exc
    source = re.sub(r"//[^\n]*|/\*.*?\*/", "", source, flags=re.S)
    statements = [part.strip() for part in source.split(";") if part.strip()]
    if not statements or not re.fullmatch(r"OPENQASM\s+2\.0", statements[0], re.I):
        raise QMutBenchParseError(f"{case_id}: missing OPENQASM 2.0 header")
    qregs: dict[str, tuple[int, int]] = {}
    cregs: dict[str, int] = {}
    operations: list[tuple[Any, ...]] = []
    for statement in statements[1:]:
        if re.fullmatch(r'include\s+"[^"\r\n]+"', statement, re.I):
            continue
        match = re.fullmatch(r"qreg\s+([A-Za-z_]\w*)\s*\[(\d+)\]", statement, re.I)
        if match:
            name, count = match.group(1), int(match.group(2))
            if count <= 0 or name in qregs:
                raise QMutBenchParseError(f"{case_id}: invalid or duplicate qreg {name!r}")
            offset = sum(size for _, size in qregs.values())
            qregs[name] = (offset, count)
            continue
        match = re.fullmatch(r"creg\s+([A-Za-z_]\w*)\s*\[(\d+)\]", statement, re.I)
        if match:
            name, count = match.group(1), int(match.group(2))
            if count <= 0 or name in cregs:
                raise QMutBenchParseError(f"{case_id}: invalid or duplicate creg {name!r}")
            cregs[name] = count
            continue
        if re.match(r"(?:OPENQASM|include|qreg|creg)\b", statement, re.I):
            raise QMutBenchParseError(f"{case_id}: malformed QASM statement {statement!r}")

        measure = re.fullmatch(r"measure\s+([A-Za-z_]\w*)\[(\d+)\]\s*->\s*([A-Za-z_]\w*)\[(\d+)\]", statement, re.I)
        if measure:
            qname, qindex, cname, cindex = measure.groups()
            if qname not in qregs or int(qindex) >= qregs[qname][1] or cname not in cregs or int(cindex) >= cregs[cname]:
                raise QMutBenchParseError(f"{case_id}: measurement references an undeclared register bit")
            operations.append(("measure", qregs[qname][0] + int(qindex)))
            continue

        measure_registers = re.fullmatch(r"measure\s+([A-Za-z_]\w*)\s*->\s*([A-Za-z_]\w*)", statement, re.I)
        if measure_registers:
            qname, cname = measure_registers.groups()
            if qname not in qregs or cname not in cregs or qregs[qname][1] != cregs[cname]:
                raise QMutBenchParseError(f"{case_id}: whole-register measurement requires declared registers of equal width")
            qoffset, qsize = qregs[qname]
            operations.extend(("measure", qoffset + idx) for idx in range(qsize))
            continue

        match = re.fullmatch(r"([A-Za-z_]\w*)(?:\s*\(([^()]*)\))?\s+(.+)", statement)
        if not match:
            raise QMutBenchParseError(f"{case_id}: malformed QASM operation {statement!r}")
        name, params_text, operand_text = match.groups()
        name = name.lower()
        params = [_qasm_expression(item.strip(), case_id) for item in params_text.split(",")] if params_text is not None else []
        qubits: list[int] = []
        for operand in operand_text.split(","):
            operand = operand.strip()
            ref = re.fullmatch(r"([A-Za-z_]\w*)\[(\d+)\]", operand)
            whole = re.fullmatch(r"([A-Za-z_]\w*)", operand)
            if ref:
                register, index = ref.group(1), int(ref.group(2))
                if register not in qregs or index >= qregs[register][1]:
                    raise QMutBenchParseError(f"{case_id}: undeclared or out-of-range qubit {operand!r}")
                qubits.append(qregs[register][0] + index)
            elif whole and name in {"barrier", "measure"} and whole.group(1) in qregs:
                offset, size = qregs[whole.group(1)]
                qubits.extend(range(offset, offset + size))
            else:
                raise QMutBenchParseError(f"{case_id}: malformed or unknown qubit operand {operand!r}")
        if name == "measure":
            operations.append(("measure", qubits))
        elif name == "barrier":
            operations.append(("barrier", *qubits))
        else:
            operations.append((name, *qubits, *params))
    if not qregs:
        raise QMutBenchParseError(f"{case_id}: QASM declares no quantum register")
    return operations, sum(size for _, size in qregs.values())


def _qasm_mutation_metadata(path: Path) -> dict[str, Any]:
    stem = path.stem
    match = re.match(r"(AddGate|RemoveGate|ReplaceGate)_(.*)", stem, re.I)
    return {"operator": match.group(1), "filename_descriptor": match.group(2), "source_filename": path.name} if match else {"source_filename": path.name}


def _discover_qmutbench_qasm(root: Path, dataset: QMutBenchDataset) -> None:
    origin_dir = root / "data" / "qmutbench" / "Origin_programs"
    mutant_root = root / "data" / "qmutbench" / "Mutated_programs"
    if not origin_dir.is_dir() or not mutant_root.is_dir():
        return
    originals = {path.stem: path for path in origin_dir.glob("*.qasm") if path.is_file()}
    for mutant_dir in sorted(path for path in mutant_root.iterdir() if path.is_dir() and path.name.startswith("Mutants_")):
        program_name = mutant_dir.name[len("Mutants_"):]
        original_path = originals.get(program_name)
        if original_path is None:
            mutant_paths = sorted(mutant_dir.rglob("*.qasm"))
            dataset.issues.append({
                "source_file": _relative(mutant_dir, root),
                "status": "UNMATCHED_MUTANT_DIRECTORY",
                "reason": f"No Origin_programs/{program_name}.qasm exists for this mutant directory.",
            })
            for mutant_path in mutant_paths:
                try:
                    mutant, num_qubits = parse_qasm_file(mutant_path)
                    parse_error = None
                except QMutBenchParseError as exc:
                    mutant, num_qubits, parse_error = None, None, str(exc)
                raw = {
                    "case_id": mutant_path.stem,
                    "mutant_circuit": mutant,
                    "num_qubits": num_qubits,
                    "mutation": _qasm_mutation_metadata(mutant_path),
                    "_qasm_unmatched_original": True,
                    "_qasm_defer_execution": True,
                }
                dataset.cases.append(QMutBenchCase(
                    mutant_path.stem, raw, [_relative(mutant_path, root)],
                    parse_error=parse_error,
                ))
                dataset.data_files.append(_relative(mutant_path, root))
                dataset.provenance_verified = True
            continue
        try:
            original, num_qubits = parse_qasm_file(original_path)
        except QMutBenchParseError as exc:
            dataset.issues.append({"source_file": _relative(original_path, root), "status": "PARSING_FAILURE", "reason": str(exc)})
            continue
        for mutant_path in sorted(mutant_dir.rglob("*.qasm")):
            try:
                mutant, mutant_qubits = parse_qasm_file(mutant_path)
                if mutant_qubits != num_qubits:
                    raise QMutBenchParseError(f"{mutant_path.name}: qubit count differs from original")
                parse_error = None
            except QMutBenchParseError as exc:
                mutant, parse_error = None, str(exc)
            source_files = [_relative(original_path, root), _relative(mutant_path, root)]
            raw = {
                "case_id": mutant_path.stem,
                "original_circuit": original,
                "mutant_circuit": mutant,
                "num_qubits": num_qubits,
                "mutation": _qasm_mutation_metadata(mutant_path),
                "_qasm_defer_execution": True,
            }
            dataset.cases.append(QMutBenchCase(mutant_path.stem, raw, source_files, parse_error=parse_error))
            dataset.data_files.extend(source_files)
            dataset.provenance_verified = True
    dataset.data_files = sorted(set(dataset.data_files))
    dataset.source_files = sorted(set(dataset.source_files + dataset.data_files))


def _qmut_path_hint(path: Path, root: Path) -> bool:
    try:
        parts = path.resolve().relative_to(root.resolve()).parts
    except ValueError:
        parts = path.parts
    return any("qmutbench" in part.lower() for part in parts)


def _dataset_shaped(document: Any) -> bool:
    rows, has_collection = _collection(document)
    return has_collection and any(_case_like(row) for row in rows)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def discover_qmutbench_files(root: str | Path) -> QMutBenchDataset:
    """Search arbitrary project filenames; only verified QMutBench provenance is accepted."""

    root_path = Path(root)
    dataset = QMutBenchDataset(root=str(root_path))
    if not root_path.exists():
        dataset.issues.append({"source_file": str(root_path), "status": "DATASET_UNAVAILABLE", "reason": "Dataset root does not exist."})
        return dataset

    for path in sorted(root_path.rglob("*")):
        if not path.is_file():
            continue
        relative = _relative(path, root_path)
        if any(part.lower() in EXCLUDED_PARTS for part in Path(relative).parts):
            continue
        suffix = path.suffix.lower()
        if suffix not in TEXT_SUFFIXES:
            continue
        path_hint = _qmut_path_hint(path, root_path)
        if suffix not in CASE_SUFFIXES:
            if path_hint:
                dataset.source_files.append(relative)
            continue
        try:
            document = _read_structured_file(path)
        except (OSError, UnicodeError, json.JSONDecodeError, csv.Error, ValueError) as exc:
            if path_hint:
                dataset.issues.append({"source_file": relative, "status": "PARSING_FAILURE", "reason": f"{type(exc).__name__}: {exc}"})
            continue

        explicit_provenance = path_hint or _contains_qmutbench_metadata(document)
        rows, has_collection = _collection(document)
        case_rows = [row for row in rows if _case_like(row)]
        if not explicit_provenance:
            continue
        dataset.provenance_verified = True
        dataset.test_fixture_data = dataset.test_fixture_data or any(
            part.lower().replace("-", "_") in {"test_fixtures", "fixtures"}
            for part in path.parts
        )
        if case_rows:
            dataset.data_files.append(relative)
            for index, raw_case in enumerate(case_rows, start=1):
                case_id = next((raw_case[key] for key in ("case_id", "benchmark_case_id", "id", "name") if raw_case.get(key) is not None), f"{path.stem}#{index}")
                dataset.cases.append(QMutBenchCase(str(case_id), raw_case, [relative]))
        elif has_collection:
            dataset.data_files.append(relative)
            dataset.issues.append({"source_file": relative, "status": "NO_CASES", "reason": "Verified QMutBench provenance, but no case-shaped entries were found."})
        else:
            dataset.metadata_files.append(relative)

    _discover_qmutbench_qasm(root_path, dataset)
    dataset.data_files = sorted(set(dataset.data_files))
    dataset.metadata_files = sorted(set(dataset.metadata_files))
    dataset.source_files = sorted(set(dataset.source_files))
    if not dataset.provenance_verified:
        dataset.issues.append({"source_file": str(root_path), "status": "DATASET_UNAVAILABLE", "reason": "No files with verified QMutBench provenance and recognizable benchmark structure were found."})
    return dataset


def load_qmutbench_dataset(root: str | Path) -> QMutBenchDataset:
    """Discover and load provenance-verified QMutBench case and metadata files."""

    return discover_qmutbench_files(root)


def _first(raw: Mapping[str, Any], names: Sequence[str]) -> Any:
    return next((raw[name] for name in names if name in raw and raw[name] is not None), None)


def _raw_circuits(raw: Mapping[str, Any]) -> tuple[Any, Any]:
    reference = _first(raw, ("reference_circuit", "original_circuit", "source_circuit", "correct_circuit", "reference_program", "original_program", "reference", "original"))
    mutant = _first(raw, ("mutant_circuit", "mutated_circuit", "mutant_program", "mutated_program", "mutant"))
    program = raw.get("program")
    if isinstance(program, Mapping):
        reference = reference if reference is not None else _first(program, ("reference", "original", "source", "correct"))
        mutant = mutant if mutant is not None else _first(program, ("mutant", "mutated", "modified"))
    return reference, mutant


def normalize_qmutbench_case(case: QMutBenchCase) -> NormalizedBenchmarkCase:
    raw = case.raw
    reference, mutant = _raw_circuits(raw)
    expected_information = {
        key: raw.get(key)
        for key in ("expected_behavior", "expected_result", "expected_output", "reference_output", "original_output", "mutant_output", "mutant_killed", "killed")
        if key in raw
    }
    mutation_metadata = raw.get("mutation") if isinstance(raw.get("mutation"), Mapping) else None
    if mutation_metadata is None:
        fields = ("mutation_operator", "operator", "mutation_type", "changed_operation", "changed_gate", "changed_qubits", "qubits_involved")
        extracted = {key: raw[key] for key in fields if key in raw}
        mutation_metadata = extracted or None

    source_fields = ("source", "source_code", "source_file", "source_location", "line", "reference_source", "citation")
    source_info = {key: raw[key] for key in source_fields if key in raw}
    references = raw.get("references", raw.get("reference_sources", raw.get("ground_truth_sources", [])))
    if isinstance(references, Mapping):
        source_references = [
            ({**dict(value), "source": key} if isinstance(value, Mapping) else {"source": key, "label": value})
            for key, value in references.items()
        ]
    elif isinstance(references, list):
        source_references = references
    elif references is None:
        source_references = []
    else:
        source_references = [references]

    normalized = NormalizedBenchmarkCase(
        benchmark_name="QMutBench",
        case_id=case.case_id,
        source_files=list(case.source_files),
        reference_circuit=reference,
        mutant_circuit=mutant,
        expected_information=expected_information or None,
        source_references=source_references,
        ground_truth=raw.get("ground_truth", raw.get("expected_result")),
        ground_truth_status="MISSING",
        mutation_metadata=dict(mutation_metadata) if mutation_metadata is not None else None,
        source_level_information=source_info,
        benchmark_metadata={key: raw[key] for key in ("benchmark_metadata", "metadata", "version") if key in raw},
        raw=raw,
    )
    normalized.ground_truth_status = determine_qmutbench_ground_truth(normalized)["status"]
    return normalized


def _truth_value(reference: Any) -> Any:
    if not isinstance(reference, Mapping):
        return reference
    return _first(reference, ("ground_truth", "benchmark_ground_truth", "mutant_killed", "killed", "outcome", "label", "expected_result", "diagnosis_category", "system_diagnosis_category"))


def _truth_entries(value: Any, default_key: str) -> list[tuple[str, Any]]:
    recognized = {"mutant_killed", "killed", "survived", "outcome", "label", "diagnosis_category", "system_diagnosis_category"}
    if isinstance(value, Mapping):
        return [(str(key), value[key]) for key in value if _normalized_key(key) in {_normalized_key(item) for item in recognized}]
    return [(default_key, value)] if value is not None else []


def determine_qmutbench_ground_truth(case: NormalizedBenchmarkCase) -> dict[str, Any]:
    raw = case.raw
    declared_truth = raw.get("ground_truth", raw.get("expected_result"))
    references = list(case.source_references)
    truth_entries = _truth_entries(declared_truth, "ground_truth")
    for key in ("mutant_killed", "killed", "survived", "outcome", "label", "diagnosis_category", "system_diagnosis_category"):
        if key in raw:
            truth_entries.append((key, raw[key]))
    reference_entries: list[tuple[str, Any]] = []
    for reference in references:
        value = _truth_value(reference)
        if value is not None:
            entries = _truth_entries(reference, "reference")
            reference_entries.extend(entries or [("reference", value)])
    grouped_values: dict[str, set[str]] = {}
    for key, value in truth_entries + reference_entries:
        grouped_values.setdefault(_normalized_key(key), set()).add(json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":")))
    conflicting = any(len(values) > 1 for values in grouped_values.values())
    if conflicting:
        status = "CONFLICTING"
    elif not truth_entries and not reference_entries and not references:
        status = "MISSING"
    elif any(
        isinstance(reference, Mapping) and reference.get("reliable") is True and reference.get("independent") is True and _truth_value(reference) is not None
        for reference in references
    ) or (declared_truth is not None and raw.get("ground_truth_reliable") is True and raw.get("reference_independent") is True):
        status = "RELIABLE"
    else:
        status = "UNCERTAIN"

    compatibility = None
    semantics = None
    if isinstance(declared_truth, Mapping):
        semantics = declared_truth.get("label_semantics")
        compatibility = declared_truth.get("system_diagnosis_category") or declared_truth.get("diagnosis_category")
    if compatibility is None:
        semantics = raw.get("label_semantics", semantics)
        compatibility = raw.get("system_diagnosis_category") or raw.get("diagnosis_category")
    return {
        "status": status,
        "value": _json_safe(declared_truth),
        "references": _json_safe(references),
        "conflicting_values": _json_safe([{"field": key, "value": value} for key, value in truth_entries + reference_entries]) if conflicting else [],
        "reliable": status == "RELIABLE",
        "diagnosis_category_compatible": semantics in {"diagnosis_category", "system_diagnosis_category"} and compatibility is not None,
        "diagnosis_category": compatibility,
        "raw_fields": {key: raw[key] for key in ("ground_truth_reliable", "reference_independent", "label_semantics") if key in raw},
    }


def _normalize_instruction(instruction: Any, case_id: str) -> tuple[Any, ...]:
    if isinstance(instruction, Mapping):
        name = str(instruction.get("name", instruction.get("op", instruction.get("gate", "")))).strip().lower()
        qubits = instruction.get("qubits", [])
        parameters = instruction.get("parameters", instruction.get("params", []))
        if isinstance(qubits, int):
            qubits = [qubits]
        if isinstance(parameters, (int, float)):
            parameters = [parameters]
        if not isinstance(qubits, (list, tuple)) or not isinstance(parameters, (list, tuple)):
            raise QMutBenchParseError(f"{case_id}: malformed operation qubits or parameters")
        instruction = (name, *qubits, *parameters)
    if not isinstance(instruction, (list, tuple)) or not instruction:
        raise QMutBenchParseError(f"{case_id}: circuit operations must be non-empty tuples, lists, or objects")

    name = str(instruction[0]).strip().lower()
    if name in CONTROLLED_PARAMETER_GATES:
        raise QMutBenchUnsupportedError(f"{case_id}: simulator does not reliably support {name!r}")
    if name not in SINGLE_GATES | ROTATION_GATES | {"u2", "u3"} | TWO_GATES | TWO_QUBIT_ROTATIONS | THREE_GATES | {"measure", "reset", "barrier"}:
        raise QMutBenchUnsupportedError(f"{case_id}: unsupported simulator operation {name!r}")
    operands = list(instruction[1:])
    parameter_count = PARAMETERIZED_ARITY.get(name, 1 if name in TWO_QUBIT_ROTATIONS else 2 if name == "u2" else 3 if name == "u3" else 0)
    if name in SINGLE_GATES:
        qubit_count = 1
    elif name in ROTATION_GATES or name in {"u2", "u3"}:
        qubit_count = 1
    elif name in TWO_GATES | TWO_QUBIT_ROTATIONS:
        qubit_count = 2
    elif name in THREE_GATES:
        qubit_count = 3
    elif name == "reset":
        qubit_count = 1
    elif name == "measure":
        if not operands:
            return ("measure",)
        if isinstance(operands[0], (list, tuple)):
            measured = tuple(operands[0])
            if not all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in measured):
                raise QMutBenchParseError(f"{case_id}: measure targets must be non-negative integer qubits")
            return ("measure", list(measured))
        qubit_count = 1
    else:
        if not all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in operands):
            raise QMutBenchParseError(f"{case_id}: barrier targets must be non-negative integer qubits")
        return ("barrier", *operands)

    if len(operands) != qubit_count + parameter_count:
        raise QMutBenchParseError(f"{case_id}: {name} expects {qubit_count} qubit(s) and {parameter_count} parameter(s)")
    qubits = operands[:qubit_count]
    if not all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in qubits):
        raise QMutBenchParseError(f"{case_id}: qubit indices must be non-negative integers")
    try:
        params = [float(value) for value in operands[qubit_count:]]
    except (TypeError, ValueError) as exc:
        raise QMutBenchParseError(f"{case_id}: gate parameters must be numeric") from exc
    if name not in {"measure", "reset", "barrier"}:
        try:
            get_gate(name)
        except (TypeError, ValueError) as exc:
            raise QMutBenchUnsupportedError(f"{case_id}: unsupported simulator operation {name!r}") from exc
    return (name, *qubits, *params)


def _normalize_circuit(value: Any, case_id: str) -> list[tuple[Any, ...]]:
    if isinstance(value, Mapping):
        value = value.get("operations", value.get("circuit"))
    if not isinstance(value, (list, tuple)):
        raise QMutBenchParseError(f"{case_id}: circuit/program must be an operation list")
    return [_normalize_instruction(operation, case_id) for operation in value]


def _circuit_qubit_count(circuit: Sequence[Sequence[Any]], declared: Any) -> int:
    if declared is not None:
        if not isinstance(declared, int) or isinstance(declared, bool) or declared <= 0:
            raise QMutBenchParseError("num_qubits must be a positive integer")
        return declared
    maximum = -1
    for operation in circuit:
        name = str(operation[0]).lower()
        if name in SINGLE_GATES or name in ROTATION_GATES or name in {"u2", "u3", "reset"}:
            qubits = operation[1:2]
        elif name in TWO_GATES | TWO_QUBIT_ROTATIONS:
            qubits = operation[1:3]
        elif name in THREE_GATES:
            qubits = operation[1:4]
        elif name == "measure":
            if len(operation) > 1 and isinstance(operation[1], (list, tuple)):
                qubits = operation[1]
            else:
                qubits = operation[1:2]
        else:
            qubits = operation[1:]
        for qubit in qubits:
            maximum = max(maximum, int(qubit))
    return max(1, maximum + 1)


def _state_distribution(circuit: Sequence[Sequence[Any]], num_qubits: int, seed: int) -> dict[str, float]:
    simulator = QuantumSimulator(num_qubits, seed=seed)
    operations = [operation for operation in circuit if str(operation[0]).lower() != "measure"]
    simulator.run_circuit(operations)
    return {f"{index:0{num_qubits}b}": float(value) for index, value in enumerate(simulator.get_probabilities())}


def _sample_measurement_counts(
    circuit: Sequence[Sequence[Any]], num_qubits: int, shots: int, seed: int,
) -> dict[str, int]:
    """Run one reproducible finite-shot sample of a circuit's full register output."""
    operations = [tuple(operation) for operation in circuit if str(operation[0]).lower() != "measure"]
    operations.append(("measure", list(range(num_qubits))))
    return QuantumSimulator(num_qubits, seed=seed).run_shots(operations, shots=shots, seed=seed)


def _counts_output(counts: Mapping[str, int], shots: int) -> dict[str, Any]:
    return {
        "counts": {str(key): int(value) for key, value in sorted(counts.items())},
        "distribution": {str(key): float(value) / shots for key, value in sorted(counts.items())},
    }


def _align_distributions(expected: Mapping[str, float], observed: Mapping[str, float]) -> tuple[dict[str, float], dict[str, float]]:
    keys = sorted(set(expected) | set(observed))
    return ({key: float(expected.get(key, 0.0)) for key in keys}, {key: float(observed.get(key, 0.0)) for key in keys})


def _candidate_payload(candidate: Any) -> dict[str, Any]:
    return _json_safe(asdict(candidate))


def validate_qmutbench_case(case: QMutBenchCase, *, threshold: float = 0.05, shots: int = 1024, seed: int = 1729) -> QMutBenchCaseResult:
    normalized = normalize_qmutbench_case(case)
    ground_truth = determine_qmutbench_ground_truth(normalized)
    empty = {
        "benchmark": "QMutBench", "case_id": case.case_id, "source_files": case.source_files,
        "support_status": "PARSING_FAILURE", "supported": False, "execution_status": "NOT_EXECUTED",
        "ground_truth_status": ground_truth["status"], "benchmark_ground_truth": ground_truth,
        "system_diagnosis": None, "diagnosis_ground_truth_compatible": False,
        "case_level_category_match": None, "reference_circuit": normalized.reference_circuit,
        "mutant_circuit": normalized.mutant_circuit, "mutation_metadata": normalized.mutation_metadata,
        "expected_information": normalized.expected_information, "source_level_information": normalized.source_level_information,
        "program_candidates": [], "hardware_candidates": [], "evidence": [], "errors": [],
        "original_program": case.source_files[0] if len(case.source_files) > 1 else None,
        "mutant_filename": Path(case.source_files[-1]).name if case.source_files else None,
        "independent_ground_truth_available": ground_truth.get("status") == "RELIABLE",
    }
    if normalized.raw.get("_qasm_unmatched_original") is True:
        empty.update({
            "support_status": "UNSUPPORTED",
            "execution_status": "NOT_EXECUTED",
            "errors": ["mutant directory has no matching original program"] + ([case.parse_error] if case.parse_error else []),
        })
        return QMutBenchCaseResult(**empty)
    if case.parse_error:
        empty["errors"] = [case.parse_error]
        return QMutBenchCaseResult(**empty)
    if normalized.reference_circuit is None or normalized.mutant_circuit is None:
        empty["errors"] = ["reference/original or mutant circuit is missing"]
        return QMutBenchCaseResult(**empty)

    try:
        reference_circuit = _normalize_circuit(normalized.reference_circuit, case.case_id)
        mutant_circuit = _normalize_circuit(normalized.mutant_circuit, case.case_id)
        declared = case.raw.get("num_qubits")
        num_qubits = max(_circuit_qubit_count(reference_circuit, declared), _circuit_qubit_count(mutant_circuit, declared))
        if any(
            qubit >= num_qubits
            for circuit in (reference_circuit, mutant_circuit)
            for operation in circuit
            for qubit in _operation_qubits(operation)
        ):
            raise QMutBenchParseError("circuit references a qubit beyond configured num_qubits")
    except QMutBenchUnsupportedError as exc:
        empty.update({"support_status": "UNSUPPORTED", "execution_status": "NOT_EXECUTED", "errors": [str(exc)]})
        return QMutBenchCaseResult(**empty)
    except (QMutBenchParseError, TypeError, ValueError, KeyError) as exc:
        empty.update({"support_status": "PARSING_FAILURE", "execution_status": "NOT_EXECUTED", "errors": [f"{type(exc).__name__}: {exc}"]})
        return QMutBenchCaseResult(**empty)

    if normalized.raw.get("_qasm_defer_execution") is True:
        reference_counts = None
        mutant_counts = None
        try:
            reference_counts = _sample_measurement_counts(reference_circuit, num_qubits, shots, seed)
            mutant_counts = _sample_measurement_counts(mutant_circuit, num_qubits, shots, seed)
            outcomes = [f"{index:0{num_qubits}b}" for index in range(2 ** num_qubits)]
            reference_distribution = {outcome: reference_counts.get(outcome, 0) / shots for outcome in outcomes}
            mutant_distribution = {outcome: mutant_counts.get(outcome, 0) / shots for outcome in outcomes}
            comparison = compare_distributions(
                reference_distribution,
                mutant_distribution,
                threshold=threshold,
                shots=shots,
            )
        except Exception as exc:
            empty.update({
                "support_status": "SUPPORTED", "supported": True,
                "execution_status": "EXECUTION_FAILURE",
                "reference_circuit": _json_safe(reference_circuit),
                "mutant_circuit": _json_safe(mutant_circuit),
                "errors": [f"{type(exc).__name__}: {exc}"],
                "original_output": _counts_output(reference_counts, shots) if reference_counts is not None else None,
                "mutant_output": _counts_output(mutant_counts, shots) if mutant_counts is not None else None,
            })
            return QMutBenchCaseResult(**empty)
        empty.update({
            "support_status": "SUPPORTED", "supported": True, "execution_status": "COMPLETED",
            "reference_circuit": _json_safe(reference_circuit), "mutant_circuit": _json_safe(mutant_circuit),
            "source_level_information": {
                **normalized.source_level_information,
                "num_qubits": num_qubits,
                "shots": shots,
                "random_seed": seed,
                "comparison_threshold": threshold,
            },
            "original_output": _counts_output(reference_counts, shots),
            "mutant_output": _counts_output(mutant_counts, shots),
            "behavioral_comparison": _json_safe(comparison),
            "behaviorally_different": bool(comparison.anomaly_detected),
            "evidence": [comparison.message],
        })
        try:
            diagnosis = diagnose_saved_qmutbench_case({
                **empty,
                "execution_status": "COMPLETED",
                "behavioral_comparison": _json_safe(comparison),
                "reference_circuit": _json_safe(reference_circuit),
                "mutant_circuit": _json_safe(mutant_circuit),
            })
            empty["system_diagnosis"] = _json_safe(diagnosis)
            empty["program_candidates"] = [_candidate_payload(candidate) for candidate in diagnosis.program_evidence_candidates]
            empty["hardware_candidates"] = [_candidate_payload(candidate) for candidate in diagnosis.hardware_evidence_candidates]
            empty["evidence"] = list(diagnosis.evidence)
        except (TypeError, ValueError, KeyError) as exc:
            empty["errors"] = [f"Diagnosis evidence unavailable: {type(exc).__name__}: {exc}"]
        return QMutBenchCaseResult(**empty)

    try:
        reference_distribution = _state_distribution(reference_circuit, num_qubits, seed)
        mutant_distribution = _state_distribution(mutant_circuit, num_qubits, seed)
        expected, observed = _align_distributions(reference_distribution, mutant_distribution)
        localizer_reference = _localization_view(reference_circuit, num_qubits)
        localizer_mutant = _localization_view(mutant_circuit, num_qubits)
        comparison = compare_distributions(expected, observed, threshold=threshold, shots=shots)
        diagnosis = diagnose_saved_qmutbench_case({
            "execution_status": "COMPLETED",
            "behavioral_comparison": _json_safe(comparison),
            "reference_circuit": _json_safe(reference_circuit),
            "mutant_circuit": _json_safe(mutant_circuit),
            "source_level_information": {"num_qubits": num_qubits},
        })
    except Exception as exc:
        empty.update({"support_status": "SUPPORTED", "supported": True, "execution_status": "EXECUTION_FAILURE", "errors": [f"{type(exc).__name__}: {exc}"]})
        return QMutBenchCaseResult(**empty)

    compatible = ground_truth["status"] == "RELIABLE" and ground_truth["diagnosis_category_compatible"]
    category_match = None
    if compatible:
        try:
            expected_category = _resolve_diagnosis_category(str(ground_truth["diagnosis_category"]))
            category_match = diagnosis.category is expected_category
        except ValueError:
            compatible = False

    source_info = dict(normalized.source_level_information)
    source_info["num_qubits"] = num_qubits
    source_info["random_seed"] = seed
    return QMutBenchCaseResult(
        benchmark="QMutBench",
        case_id=case.case_id,
        source_files=list(case.source_files),
        support_status="SUPPORTED",
        supported=True,
        execution_status="COMPLETED",
        ground_truth_status=ground_truth["status"],
        benchmark_ground_truth=ground_truth,
        system_diagnosis=_json_safe(diagnosis),
        diagnosis_ground_truth_compatible=bool(compatible),
        case_level_category_match=category_match,
        reference_circuit=_json_safe(reference_circuit),
        mutant_circuit=_json_safe(mutant_circuit),
        mutation_metadata=_json_safe(normalized.mutation_metadata),
        expected_information=_json_safe(normalized.expected_information),
        source_level_information=_json_safe(source_info),
        program_candidates=[_candidate_payload(candidate) for candidate in diagnosis.program_evidence_candidates],
        hardware_candidates=[_candidate_payload(candidate) for candidate in diagnosis.hardware_evidence_candidates],
        evidence=list(diagnosis.evidence),
        errors=[],
        behavioral_comparison=_json_safe(comparison),
        behaviorally_different=bool(comparison.anomaly_detected),
        independent_ground_truth_available=ground_truth.get("status") == "RELIABLE",
    )


def _operation_qubits(operation: Sequence[Any]) -> list[int]:
    name = str(operation[0]).lower()
    if name in SINGLE_GATES or name in ROTATION_GATES or name in {"u2", "u3", "reset"}:
        return [int(operation[1])] if len(operation) > 1 else []
    if name in TWO_GATES:
        return [int(value) for value in operation[1:3]]
    if name in TWO_QUBIT_ROTATIONS:
        return [int(value) for value in operation[1:3]]
    if name in THREE_GATES:
        return [int(value) for value in operation[1:4]]
    if name == "measure":
        if len(operation) > 1 and isinstance(operation[1], (list, tuple)):
            return [int(value) for value in operation[1]]
        return [int(operation[1])] if len(operation) > 1 else []
    if name == "barrier":
        return [int(value) for value in operation[1:]]
    return []


def _localization_view(circuit: Sequence[Sequence[Any]], num_qubits: int) -> list[tuple[Any, ...]]:
    result = []
    for operation in circuit:
        if str(operation[0]).lower() == "measure":
            qubits = _operation_qubits(operation) or list(range(num_qubits))
            result.append(("barrier", *qubits))
        else:
            result.append(tuple(operation))
    return result


def diagnose_saved_qmutbench_case(case: Mapping[str, Any]) -> DiagnosisResult:
    """Run the existing diagnosis pipeline on one completed saved report case.

    The full distributions in ``behavioral_comparison`` are used directly;
    sparse count-derived outputs are intentionally not used here.
    """
    if not isinstance(case, Mapping):
        raise TypeError("case must be a mapping from a saved QMutBench report")
    if case.get("execution_status") != "COMPLETED":
        raise ValueError("saved QMutBench case must have execution_status COMPLETED")
    comparison = case.get("behavioral_comparison")
    if not isinstance(comparison, Mapping):
        raise ValueError("saved QMutBench case is missing behavioral_comparison")
    expected_distribution = comparison.get("expected_distribution")
    observed_distribution = comparison.get("observed_distribution")
    if not isinstance(expected_distribution, Mapping) or not isinstance(observed_distribution, Mapping):
        raise ValueError("behavioral_comparison must contain full expected and observed distributions")
    reference_circuit = case.get("reference_circuit")
    mutant_circuit = case.get("mutant_circuit")
    if not isinstance(reference_circuit, Sequence) or not isinstance(mutant_circuit, Sequence):
        raise ValueError("saved QMutBench case must contain reference_circuit and mutant_circuit")
    source_info = case.get("source_level_information") or {}
    declared_qubits = source_info.get("num_qubits") if isinstance(source_info, Mapping) else None
    num_qubits = _circuit_qubit_count(reference_circuit, declared_qubits)
    num_qubits = max(num_qubits, _circuit_qubit_count(mutant_circuit, declared_qubits))
    return diagnose_execution(
        expected_distribution=expected_distribution,
        observed_distribution=observed_distribution,
        expected_circuit=_localization_view(reference_circuit, num_qubits),
        observed_circuit=_localization_view(mutant_circuit, num_qubits),
        threshold=0.05,
        shots=1024,
    )


def _resolve_diagnosis_category(value: str) -> DiagnosisCategory:
    normalized = value.strip().upper()
    if normalized in {"AMBIGUOUS", "INSUFFICIENT_EVIDENCE", "AMBIGUOUS / INSUFFICIENT_EVIDENCE"}:
        return DiagnosisCategory.AMBIGUOUS
    try:
        return DiagnosisCategory[normalized]
    except KeyError as exc:
        raise ValueError(f"Unknown diagnosis category {value!r}") from exc


def validate_qmutbench_dataset(dataset: QMutBenchDataset) -> QMutBenchValidationReport:
    results = [validate_qmutbench_case(case) for case in dataset.cases]
    if not dataset.provenance_verified:
        status = "PARTIAL / DATASET_UNAVAILABLE"
    elif not dataset.cases:
        status = "PARTIAL / NO_CASES"
    else:
        status = "EVALUATED"
    return QMutBenchValidationReport(
        root=dataset.root,
        dataset_found=dataset.provenance_verified,
        status=status,
        data_files=list(dataset.data_files),
        metadata_files=list(dataset.metadata_files),
        source_files=list(dataset.source_files),
        provenance_verified=dataset.provenance_verified,
        cases=results,
        issues=list(dataset.issues),
        test_fixture_data=dataset.test_fixture_data,
    )


def run_qmutbench_validation(root: str | Path, output_path: str | Path | None = None) -> QMutBenchValidationReport:
    report = validate_qmutbench_dataset(discover_qmutbench_files(root))
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


class QMutBenchAdapter(BenchmarkAdapter):
    benchmark_name = "QMutBench"

    def run(self, root: str | Path) -> Mapping[str, Any]:
        return run_qmutbench_validation(root).to_dict()


__all__ = [
    "QMutBenchAdapter",
    "QMutBenchCase",
    "QMutBenchCaseResult",
    "QMutBenchDataset",
    "QMutBenchValidationReport",
    "determine_qmutbench_ground_truth",
    "discover_qmutbench_files",
    "load_qmutbench_dataset",
    "normalize_qmutbench_case",
    "run_qmutbench_validation",
    "validate_qmutbench_case",
    "validate_qmutbench_dataset",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate discovered QMutBench cases without injecting mutation labels into diagnosis.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent / "reports" / "benchmarks" / "qmutbench_validation_report.json"))
    args = parser.parse_args()
    report = run_qmutbench_validation(args.root, args.output)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
