"""Transparent Bugs4Q dataset adapter for the existing diagnosis pipeline.

This module contains no case-specific answers or diagnosis rules. It accepts a
small, documented JSON/JSONL/CSV interchange shape, retains raw reference data,
and reports missing datasets, unsupported instructions, and execution failures
without converting them into diagnoses.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from diagnosis_engine import DiagnosisCategory, diagnose_execution
from gates import get_gate
from hardware_model import HardwareModel


SUPPORTED_SUFFIXES = {".json", ".jsonl", ".ndjson", ".csv"}
SINGLE_QUBIT_GATES = {"h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx", "sxdg"}
PARAMETERIZED_GATES = {"rx", "ry", "rz", "p", "u1", "cp", "crx", "cry", "crz", "u2", "u3"}
TWO_QUBIT_GATES = {"cx", "cnot", "cz", "swap", "ch"}
THREE_QUBIT_GATES = {"ccx", "toffoli", "cswap", "fredkin"}
SPECIAL_OPERATIONS = {"measure", "reset", "barrier"}
CONTROLLED_PARAMETERIZED_GATES = {"cp", "crx", "cry", "crz"}


class CaseParseError(ValueError):
    """A case is malformed or cannot be represented by the current pipeline."""


@dataclass
class DatasetIssue:
    source_file: str
    reason: str
    status: str = "DATASET_ISSUE"

    def to_dict(self) -> dict[str, str]:
        return {"source_file": self.source_file, "reason": self.reason, "status": self.status}


@dataclass
class Bugs4QCase:
    case_id: str
    raw: Mapping[str, Any]
    source_file: str
    parse_error: str | None = None


@dataclass
class Bugs4QDataset:
    root: str
    files: list[str] = field(default_factory=list)
    cases: list[Bugs4QCase] = field(default_factory=list)
    issues: list[DatasetIssue] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return bool(self.files)


@dataclass
class Bugs4QCaseResult:
    case_id: str
    source_file: str
    supported: bool
    support_status: str
    reference_status: str
    expected_reference: dict[str, Any]
    observed_category: DiagnosisCategory | None
    diagnosis_summary: str | None
    program_candidates: list[dict[str, Any]]
    hardware_candidates: list[dict[str, Any]]
    evidence: list[str]
    validation_status: str
    case_level_category_match: bool | None
    unsupported_or_error_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "source_file": self.source_file,
            "supported": self.supported,
            "support_status": self.support_status,
            "expected_reference": self.expected_reference,
            "observed_category": self.observed_category.value if self.observed_category else None,
            "diagnosis_summary": self.diagnosis_summary,
            "program_fault_candidates": self.program_candidates,
            "hardware_fault_candidates": self.hardware_candidates,
            "evidence": list(self.evidence),
            "validation_status": self.validation_status,
            "case_level_category_match": self.case_level_category_match,
            "unsupported_or_error_reason": self.unsupported_or_error_reason,
        }


@dataclass
class Bugs4QValidationReport:
    root: str
    dataset_found: bool
    discovered_files: list[str] = field(default_factory=list)
    records: list[Bugs4QCaseResult] = field(default_factory=list)
    dataset_issues: list[DatasetIssue] = field(default_factory=list)

    @property
    def discovered_cases(self) -> int:
        return len(self.records)

    def count_status(self, status: str) -> int:
        return sum(record.support_status == status for record in self.records)

    @property
    def reliable_ground_truth_cases(self) -> int:
        return sum(record.reference_status == "RELIABLE" and record.supported for record in self.records)

    @property
    def uncertain_ground_truth_cases(self) -> int:
        return sum(record.reference_status in {"UNCERTAIN", "CONFLICTING", "MISSING"} for record in self.records)

    @property
    def supported_cases(self) -> int:
        return sum(record.supported for record in self.records)

    @property
    def unsupported_cases(self) -> int:
        return self.count_status("UNSUPPORTED")

    @property
    def execution_failures(self) -> int:
        case_failures = self.count_status("EXECUTION_FAILURE") + self.count_status("PARSING_FAILURE")
        file_parse_failures = sum(issue.status == "PARSING_FAILURE" for issue in self.dataset_issues)
        return case_failures + file_parse_failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": "Bugs4Q",
            "dataset_found": self.dataset_found,
            "dataset_root": self.root,
            "discovered_files": list(self.discovered_files),
            "cases_discovered": self.discovered_cases,
            "supported": self.supported_cases,
            "unsupported": self.unsupported_cases,
            "execution_or_parsing_failures": self.execution_failures,
            "dataset_file_parse_failures": sum(issue.status == "PARSING_FAILURE" for issue in self.dataset_issues),
            "reliable_ground_truth_cases": self.reliable_ground_truth_cases,
            "uncertain_or_conflicting_ground_truth_cases": self.uncertain_ground_truth_cases,
            "dataset_issues": [issue.to_dict() for issue in self.dataset_issues],
            "case_level_results": [record.to_dict() for record in self.records],
            "metrics": {
                "scope": "case-level only; no aggregate benchmark metrics",
                "category_matches": [
                    {"case_id": record.case_id, "match": record.case_level_category_match}
                    for record in self.records
                    if record.case_level_category_match is not None
                ],
                "aggregate_accuracy": None,
                "aggregate_precision": None,
                "aggregate_recall": None,
                "aggregate_f1": None,
            },
            "notes": [
                "Synthetic validation fixtures are not Bugs4Q cases.",
                "Only explicitly independent, reliable, non-conflicting references are eligible for case-level category comparison.",
                "No diagnosis is forced for unsupported cases or cases with uncertain references.",
            ],
        }


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def discover_bugs4q_files(root: str | Path) -> tuple[list[Path], list[Path]]:
    """Find Bugs4Q-named files and separately return recognized-path unsupported formats."""

    root_path = Path(root)
    if root_path.is_file():
        return ([root_path] if root_path.suffix.lower() in SUPPORTED_SUFFIXES else []), (
            [root_path] if root_path.suffix.lower() not in SUPPORTED_SUFFIXES else []
        )
    if not root_path.exists():
        return [], []

    supported: list[Path] = []
    unsupported: list[Path] = []
    for path in root_path.rglob("*"):
        if not path.is_file() or any(part.lower() in {".venv", ".git", "__pycache__", ".pytest_cache"} for part in path.parts):
            continue
        if path.name.lower().startswith("bugs4q_validation_report"):
            continue
        if path.suffix.lower() in {".py", ".pyc"}:
            continue
        if not any("bugs4q" in part.lower() for part in path.parts):
            continue
        if path.suffix.lower() in SUPPORTED_SUFFIXES:
            supported.append(path)
        else:
            unsupported.append(path)
    return sorted(supported), sorted(unsupported)


def _decode_csv_value(value: str) -> Any:
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _case_objects(document: Any, source_file: str) -> list[tuple[str, Mapping[str, Any]]]:
    if isinstance(document, list):
        objects = document
        ids: list[str | None] = [None] * len(objects)
    elif isinstance(document, Mapping):
        collection_key = next((key for key in ("cases", "instances", "records", "bugs4q_cases") if key in document), None)
        if collection_key is not None:
            collection = document[collection_key]
            if isinstance(collection, list):
                objects = collection
                ids = [None] * len(objects)
            elif isinstance(collection, Mapping):
                objects = list(collection.values())
                ids = [str(key) for key in collection]
            else:
                raise CaseParseError(f"{collection_key} must contain a list or object of cases")
        elif any(key in document for key in ("case_id", "id", "bug_id")):
            objects, ids = [document], [None]
        else:
            objects, ids = list(document.values()), [str(key) for key in document]
    else:
        raise CaseParseError("Dataset document must be a case, list, or object of cases")

    result: list[tuple[str, Mapping[str, Any]]] = []
    for index, (item, fallback_id) in enumerate(zip(objects, ids), start=1):
        generated_id = fallback_id or f"{Path(source_file).stem}#{index}"
        if not isinstance(item, Mapping):
            result.append((generated_id, {"_parse_error": "Case entry must be an object."}))
            continue
        case_id = next((item[key] for key in ("case_id", "id", "bug_id", "name") if item.get(key) is not None), generated_id)
        result.append((str(case_id), item))
    return result


def _read_dataset_file(path: Path, root: Path) -> tuple[list[Bugs4QCase], list[DatasetIssue]]:
    relative = _relative(path, root)
    try:
        suffix = path.suffix.lower()
        if suffix == ".json":
            document = json.loads(path.read_text(encoding="utf-8"))
            objects = _case_objects(document, relative)
        elif suffix in {".jsonl", ".ndjson"}:
            objects = []
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    document = json.loads(line)
                    objects.extend(_case_objects([document], f"{Path(relative).stem}#{line_number}"))
                except (json.JSONDecodeError, CaseParseError) as exc:
                    objects.append((f"{Path(path).stem}#{line_number}", {"_parse_error": str(exc)}))
        else:
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                objects = []
                for index, row in enumerate(csv.DictReader(stream), start=1):
                    decoded = {key: _decode_csv_value(value) for key, value in row.items() if key is not None}
                    objects.extend(_case_objects([decoded], f"{Path(relative).stem}#{index}"))

        cases = [Bugs4QCase(case_id=case_id, raw=raw, source_file=relative, parse_error=raw.get("_parse_error")) for case_id, raw in objects]
        return cases, []
    except (OSError, UnicodeError, json.JSONDecodeError, csv.Error, CaseParseError) as exc:
        return [], [DatasetIssue(relative, f"{type(exc).__name__}: {exc}", "PARSING_FAILURE")]


def load_bugs4q_dataset(root: str | Path, paths: Iterable[str | Path] | None = None) -> Bugs4QDataset:
    """Load JSON, JSONL/NDJSON, or CSV cases; retain parse issues in the report."""

    root_path = Path(root)
    if paths is None:
        supported, unsupported = discover_bugs4q_files(root_path)
    else:
        supplied = [Path(path) for path in paths]
        supported = [path for path in supplied if path.suffix.lower() in SUPPORTED_SUFFIXES]
        unsupported = [path for path in supplied if path.suffix.lower() not in SUPPORTED_SUFFIXES]

    files = [_relative(path, root_path if root_path.is_dir() else root_path.parent) for path in supported + unsupported]
    cases: list[Bugs4QCase] = []
    issues = [DatasetIssue(
        _relative(path, root_path if root_path.is_dir() else root_path.parent),
        f"Unsupported dataset file format: {path.suffix or '<no extension>'}",
        "UNSUPPORTED_FORMAT",
    ) for path in unsupported]
    for path in supported:
        loaded_cases, file_issues = _read_dataset_file(path, root_path if root_path.is_dir() else root_path.parent)
        cases.extend(loaded_cases)
        issues.extend(file_issues)

    if not files:
        issues.append(DatasetIssue(str(root_path), "No Bugs4Q dataset files were found.", "MISSING_DATASET"))
    return Bugs4QDataset(root=str(root_path), files=files, cases=cases, issues=issues)


def _first(raw: Mapping[str, Any], names: Sequence[str]) -> Any:
    return next((raw[name] for name in names if name in raw and raw[name] is not None), None)


def _normalize_instruction(instruction: Any, case_id: str) -> list[tuple[Any, ...]]:
    if isinstance(instruction, Mapping):
        name = str(_first(instruction, ("name", "op", "gate")) or "").strip().lower()
        qubits = instruction.get("qubits", [])
        parameters = instruction.get("parameters", instruction.get("params", []))
        if isinstance(qubits, int):
            qubits = [qubits]
        if not isinstance(qubits, (list, tuple)) or not isinstance(parameters, (list, tuple)):
            raise CaseParseError(f"{case_id}: operation {name!r} has malformed qubits or parameters")
        instruction = (name, *qubits, *parameters)

    if not isinstance(instruction, (list, tuple)) or not instruction:
        raise CaseParseError(f"{case_id}: every circuit instruction must be a non-empty list, tuple, or operation object")

    name = str(instruction[0]).strip().lower()
    operands = list(instruction[1:])
    if name in CONTROLLED_PARAMETERIZED_GATES:
        raise NotImplementedError(f"Unsupported simulator operation {name!r}: simulator dispatch does not support this gate reliably")
    if name not in SPECIAL_OPERATIONS | SINGLE_QUBIT_GATES | PARAMETERIZED_GATES | TWO_QUBIT_GATES | THREE_QUBIT_GATES:
        raise NotImplementedError(f"Unsupported simulator operation {name!r}")

    if name in SINGLE_QUBIT_GATES:
        qubit_count, parameter_count = 1, 0
    elif name in PARAMETERIZED_GATES:
        qubit_count = 1
        parameter_count = 2 if name == "u2" else 3 if name == "u3" else 1
    elif name in TWO_QUBIT_GATES:
        qubit_count, parameter_count = 2, 0
    elif name in THREE_QUBIT_GATES:
        qubit_count, parameter_count = 3, 0
    elif name == "reset":
        qubit_count, parameter_count = 1, 0
    elif name == "measure":
        if not operands:
            return [("measure",)]
        target = operands[0]
        if isinstance(target, (list, tuple)):
            qubits = list(target)
            if not all(isinstance(qubit, int) and not isinstance(qubit, bool) for qubit in qubits):
                raise CaseParseError(f"{case_id}: measurement qubits must be integers")
            return [("measure", qubit) for qubit in qubits] if qubits else [("measure",)]
        qubit_count, parameter_count = 1, 0
    else:
        if not all(isinstance(qubit, int) and not isinstance(qubit, bool) for qubit in operands):
            raise CaseParseError(f"{case_id}: barrier operands must be integer qubit indices")
        return [("barrier", *operands)]

    if len(operands) != qubit_count + parameter_count:
        raise CaseParseError(
            f"{case_id}: {name!r} expects {qubit_count} qubit operand(s) and {parameter_count} parameter(s), got {len(operands)} operand(s)"
        )
    qubits = operands[:qubit_count]
    parameters = operands[qubit_count:]
    if not all(isinstance(qubit, int) and not isinstance(qubit, bool) and qubit >= 0 for qubit in qubits):
        raise CaseParseError(f"{case_id}: qubit operands must be non-negative integers")
    try:
        normalized_parameters = [float(value) for value in parameters]
    except (TypeError, ValueError) as exc:
        raise CaseParseError(f"{case_id}: gate parameters must be numeric") from exc
    try:
        get_gate(name)
    except (TypeError, ValueError) as exc:
        raise NotImplementedError(f"Unsupported simulator operation {name!r}") from exc
    return [(name, *qubits, *normalized_parameters)]


def _normalize_circuit(circuit: Any, case_id: str) -> list[tuple[Any, ...]]:
    if not isinstance(circuit, (list, tuple)):
        raise CaseParseError(f"{case_id}: circuit/program must be a list or tuple of operations")
    normalized: list[tuple[Any, ...]] = []
    for instruction in circuit:
        normalized.extend(_normalize_instruction(instruction, case_id))
    return normalized


def _reference_info(raw: Mapping[str, Any]) -> tuple[dict[str, Any], DiagnosisCategory | None, str]:
    references = _first(raw, ("references", "reference_sources", "ground_truth_sources", "ground_truth"))
    if references is None:
        category = _first(raw, ("reference_category", "expected_category", "ground_truth_category"))
        source = _first(raw, ("reference_source", "ground_truth_source"))
        reliable = raw.get("ground_truth_reliable") is True
        independent = raw.get("reference_independent") is True
        refs = [] if category is None else [{
            "source": source,
            "category": category,
            "reliable": reliable,
            "independent": independent,
        }]
    elif isinstance(references, Mapping):
        refs = [dict(value, source=key) if isinstance(value, Mapping) else {"source": key, "category": value} for key, value in references.items()]
    elif isinstance(references, list):
        refs = references
    else:
        refs = [{"raw": references}]

    normalized_references: list[dict[str, Any]] = []
    categories: list[DiagnosisCategory] = []
    declared_categories: list[str] = []
    for reference in refs:
        if not isinstance(reference, Mapping):
            normalized_references.append({"raw": reference})
            continue
        category_value = _first(reference, ("category", "diagnosis_category", "label", "expected_category"))
        category_text = str(category_value) if category_value is not None else None
        if category_text:
            declared_categories.append(category_text)
            try:
                category = _resolve_diagnosis_category(category_text)
            except ValueError:
                category = None
            if category is not None:
                categories.append(category)
        normalized_references.append({
            "source": reference.get("source", reference.get("name")),
            "category": category_text,
            "reliable": reference.get("reliable") is True,
            "independent": reference.get("independent") is True,
            "raw": dict(reference),
        })

    reliable_categories = [
        _resolve_diagnosis_category(reference["category"])
        for reference in normalized_references
        if reference.get("category") and reference.get("reliable") and reference.get("independent")
        and _category_is_recognized(reference["category"])
    ]
    category_tokens = {category.value for category in categories}
    conflict = len(category_tokens) > 1
    reference_category = reliable_categories[0] if reliable_categories and len({c.value for c in reliable_categories}) == 1 and not conflict else None
    if conflict:
        status = "CONFLICTING"
    elif reference_category is not None:
        status = "RELIABLE"
    else:
        status = "UNCERTAIN" if normalized_references else "MISSING"

    return {
        "references": normalized_references,
        "declared_categories": declared_categories,
        "conflicting": conflict,
        "ground_truth_reliable": status == "RELIABLE",
        "raw_reference_fields": {
            key: value for key, value in raw.items()
            if any(marker in key.lower() for marker in ("reference", "ground_truth", "expected_answer"))
            and key not in {"references", "reference_sources", "ground_truth_sources", "ground_truth"}
        },
    }, reference_category, status


def _category_is_recognized(value: str) -> bool:
    try:
        _resolve_diagnosis_category(value)
        return True
    except ValueError:
        return False


def _resolve_diagnosis_category(value: str) -> DiagnosisCategory:
    normalized = value.strip().upper()
    if normalized in {"AMBIGUOUS", "INSUFFICIENT_EVIDENCE", "AMBIGUOUS / INSUFFICIENT_EVIDENCE"}:
        return DiagnosisCategory.AMBIGUOUS
    try:
        return DiagnosisCategory[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported reference diagnosis category: {value!r}") from exc


def _hardware_model(raw: Mapping[str, Any], n_qubits: int) -> HardwareModel | None:
    data = _first(raw, ("hardware_model", "hardware_calibration", "calibration"))
    if data is None:
        return None
    if not isinstance(data, Mapping):
        raise CaseParseError("hardware calibration must be an object")

    count = data.get("n_qubits", n_qubits)
    if not isinstance(count, int) or count <= 0:
        raise CaseParseError("hardware calibration n_qubits must be a positive integer")
    coupling = data.get("coupling_map", data.get("connectivity", []))
    model = HardwareModel(count, coupling_map=coupling, backend_name=str(data.get("backend_name", "generic_backend")))

    qubits = data.get("qubits", {})
    if isinstance(qubits, Mapping):
        for index, properties in qubits.items():
            if not isinstance(properties, Mapping):
                raise CaseParseError(f"qubit calibration {index!r} must be an object")
            model.set_qubit_properties(int(index), **{key: properties[key] for key in ("t1", "t2", "readout_error", "status", "available") if key in properties})

    edges = data.get("edges", {})
    if isinstance(edges, list):
        edge_entries = edges
    elif isinstance(edges, Mapping):
        edge_entries = [dict(properties, qubits=pair) for pair, properties in edges.items()]
    else:
        raise CaseParseError("edge calibration must be a list or object")
    for edge in edge_entries:
        if not isinstance(edge, Mapping):
            raise CaseParseError("edge calibration entries must be objects")
        pair = edge.get("qubits", edge.get("pair"))
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise CaseParseError("edge calibration requires two qubit indices")
        if tuple(sorted(pair)) not in [tuple(sorted(existing)) for existing in model.get_coupling_map()]:
            model.add_edge(int(pair[0]), int(pair[1]), **{key: edge[key] for key in ("direction", "error", "duration", "supported") if key in edge})

    gates = data.get("gates", {})
    gate_entries: list[tuple[str, Mapping[str, Any]]] = []
    if isinstance(gates, Mapping):
        for name, entries in gates.items():
            if isinstance(entries, Mapping):
                entries = [entries]
            if not isinstance(entries, list):
                raise CaseParseError(f"gate calibration {name!r} must be an object or list")
            gate_entries.extend((str(name), entry) for entry in entries)
    elif isinstance(gates, list):
        gate_entries.extend((str(entry.get("gate", entry.get("name", ""))), entry) for entry in gates if isinstance(entry, Mapping))
    else:
        raise CaseParseError("gate calibration must be a list or object")
    for name, entry in gate_entries:
        if not isinstance(entry, Mapping):
            raise CaseParseError(f"gate calibration {name!r} entry must be an object")
        model.add_gate_calibration(
            name,
            qubits=entry.get("qubits", []),
            **{key: entry[key] for key in ("error", "duration", "supported") if key in entry},
        )
    return model


def _candidate_dict(candidate: Any, source: str) -> dict[str, Any]:
    if source == "program":
        return {
            "index": candidate.index,
            "operation": candidate.operation,
            "qubits": list(candidate.qubits),
            "score": candidate.score,
            "reason": candidate.reason,
            "evidence": list(candidate.evidence),
        }
    component = list(candidate.component) if isinstance(candidate.component, tuple) else candidate.component
    return {
        "component_type": candidate.component_type,
        "component": component,
        "score": candidate.score,
        "reason": candidate.reason,
        "evidence": list(candidate.evidence),
    }


def _result_for_error(case: Bugs4QCase, status: str, reason: str, reference_info: dict[str, Any], reference_status: str) -> Bugs4QCaseResult:
    return Bugs4QCaseResult(
        case_id=case.case_id,
        source_file=case.source_file,
        supported=False,
        support_status=status,
        reference_status=reference_status,
        expected_reference=reference_info,
        observed_category=None,
        diagnosis_summary=None,
        program_candidates=[],
        hardware_candidates=[],
        evidence=[],
        validation_status="NOT_ASSESSED",
        case_level_category_match=None,
        unsupported_or_error_reason=reason,
    )


def validate_bugs4q_case(case: Bugs4QCase) -> Bugs4QCaseResult:
    """Run one parsed case through the unchanged diagnosis engine."""

    if case.parse_error:
        info, _, status = _reference_info(case.raw)
        return _result_for_error(case, "PARSING_FAILURE", case.parse_error, info, status)

    reference_info, reference_category, reference_status = _reference_info(case.raw)
    raw = case.raw
    try:
        program = raw.get("program")
        expected_circuit = _first(raw, ("expected_circuit", "ideal_circuit", "reference_circuit"))
        observed_circuit = _first(raw, ("observed_circuit", "actual_circuit", "buggy_circuit", "faulty_circuit"))
        if isinstance(program, Mapping):
            expected_circuit = expected_circuit or _first(program, ("expected", "ideal", "reference"))
            observed_circuit = observed_circuit or _first(program, ("observed", "actual", "buggy", "faulty"))
            if expected_circuit is None and observed_circuit is None:
                expected_circuit = observed_circuit = _first(program, ("circuit", "operations"))
        if expected_circuit is None and observed_circuit is None:
            single_circuit = _first(raw, ("circuit", "operations", "program"))
            if single_circuit is not None and not isinstance(single_circuit, Mapping):
                expected_circuit = observed_circuit = single_circuit
        if expected_circuit is None:
            expected_circuit = observed_circuit
        if observed_circuit is None:
            observed_circuit = expected_circuit
        if expected_circuit is None:
            raise CaseParseError("No circuit/program field is available")

        expected_ops = _normalize_circuit(expected_circuit, case.case_id)
        observed_ops = _normalize_circuit(observed_circuit, case.case_id)
        all_qubits = [qubit for operation in expected_ops + observed_ops for qubit in _instruction_qubits(operation)]
        inferred_qubits = max(all_qubits, default=-1) + 1
        hardware_model = _hardware_model(raw, max(inferred_qubits, 1))
        expected_distribution = _first(raw, ("expected_distribution", "ideal_distribution", "expected_counts", "ideal_counts"))
        observed_distribution = _first(raw, ("observed_distribution", "actual_distribution", "observed_counts", "actual_counts", "counts"))
        threshold = raw.get("threshold", 0.05)
        shots = raw.get("shots")
        if hardware_model is not None and any(qubit >= hardware_model.n_qubits for qubit in all_qubits):
            raise CaseParseError("Circuit references a qubit outside the supplied hardware calibration range")
    except NotImplementedError as exc:
        return _result_for_error(case, "UNSUPPORTED", str(exc), reference_info, reference_status)
    except (CaseParseError, TypeError, ValueError, KeyError) as exc:
        return _result_for_error(case, "PARSING_FAILURE", f"{type(exc).__name__}: {exc}", reference_info, reference_status)

    try:
        result = diagnose_execution(
            expected_distribution=expected_distribution,
            observed_distribution=observed_distribution,
            expected_circuit=expected_ops,
            observed_circuit=observed_ops,
            hardware_model=hardware_model,
            threshold=threshold,
            shots=shots,
        )
    except Exception as exc:
        return _result_for_error(case, "EXECUTION_FAILURE", f"{type(exc).__name__}: {exc}", reference_info, reference_status)

    support_status = "SUPPORTED_VERIFIABLE" if reference_status == "RELIABLE" else "SUPPORTED_UNCERTAIN"
    match = None
    validation_status = "NOT_ASSESSED"
    if reference_category is not None and reference_status == "RELIABLE":
        match = result.category is reference_category
        validation_status = "MATCH" if match else "MISMATCH"

    evidence = [result.summary, *result.evidence]
    if expected_circuit == observed_circuit and raw.get("expected_circuit") is None and raw.get("observed_circuit") is None:
        evidence.append("Only one circuit/program was supplied and used for both expected and observed circuit inputs.")
    if reference_status in {"UNCERTAIN", "CONFLICTING", "MISSING"}:
        evidence.append(f"Reference status is {reference_status.lower()}; this case is not counted as confirmed ground truth.")

    return Bugs4QCaseResult(
        case_id=case.case_id,
        source_file=case.source_file,
        supported=True,
        support_status=support_status,
        reference_status=reference_status,
        expected_reference=reference_info,
        observed_category=result.category,
        diagnosis_summary=result.summary,
        program_candidates=[_candidate_dict(candidate, "program") for candidate in result.program_evidence_candidates],
        hardware_candidates=[_candidate_dict(candidate, "hardware") for candidate in result.hardware_evidence_candidates],
        evidence=evidence,
        validation_status=validation_status,
        case_level_category_match=match,
    )


def _instruction_qubits(instruction: Sequence[Any]) -> list[int]:
    name = str(instruction[0]).lower()
    if name == "measure" and len(instruction) > 1 and isinstance(instruction[1], (list, tuple)):
        return [int(qubit) for qubit in instruction[1]]
    if name == "barrier":
        return [int(qubit) for qubit in instruction[1:]]
    if name in SINGLE_QUBIT_GATES or name in PARAMETERIZED_GATES or name == "reset":
        return [int(instruction[1])] if len(instruction) > 1 else []
    if name in TWO_QUBIT_GATES:
        return [int(qubit) for qubit in instruction[1:3]]
    if name in THREE_QUBIT_GATES:
        return [int(qubit) for qubit in instruction[1:4]]
    if name == "measure" and len(instruction) > 1:
        return [int(instruction[1])]
    return []


def validate_bugs4q_dataset(dataset: Bugs4QDataset) -> Bugs4QValidationReport:
    records = [validate_bugs4q_case(case) for case in dataset.cases]
    return Bugs4QValidationReport(
        root=dataset.root,
        dataset_found=dataset.found,
        discovered_files=list(dataset.files),
        records=records,
        dataset_issues=list(dataset.issues),
    )


def write_bugs4q_report(report: Bugs4QValidationReport, output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def run_bugs4q_validation(root: str | Path, output_path: str | Path | None = None) -> Bugs4QValidationReport:
    dataset = load_bugs4q_dataset(root)
    report = validate_bugs4q_dataset(dataset)
    if output_path is not None:
        write_bugs4q_report(report, output_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the existing diagnosis pipeline against discovered Bugs4Q cases.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent), help="Project or dataset root to search")
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent / "reports" / "bugs4q_validation_report.json"))
    args = parser.parse_args()
    report = run_bugs4q_validation(args.root, args.output)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0


__all__ = [
    "Bugs4QCase",
    "Bugs4QCaseResult",
    "Bugs4QDataset",
    "Bugs4QValidationReport",
    "DatasetIssue",
    "discover_bugs4q_files",
    "load_bugs4q_dataset",
    "validate_bugs4q_case",
    "validate_bugs4q_dataset",
    "run_bugs4q_validation",
    "write_bugs4q_report",
]


if __name__ == "__main__":
    raise SystemExit(main())
