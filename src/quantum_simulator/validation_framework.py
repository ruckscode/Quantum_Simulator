"""Backend-independent validation scenarios for the diagnosis pipeline.

The default scenarios are controlled synthetic fixtures, not benchmark results.
The framework evaluates explicit expected categories and retains the diagnosis
engine's outputs and supporting evidence. It does not calculate accuracy metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .diagnosis_engine import DiagnosisCategory, DiagnosisResult, diagnose_execution
from .hardware_model import HardwareModel


@dataclass
class ValidationCase:
    case_id: str
    expected_category: DiagnosisCategory | str
    expected_circuit: Sequence[Any] = field(default_factory=tuple)
    observed_circuit: Sequence[Any] | None = None
    expected_distribution: Mapping[str, float] | Sequence[float] | None = None
    observed_distribution: Mapping[str, float] | Sequence[float] | None = None
    hardware_model: HardwareModel | None = None
    threshold: float = 0.05
    shots: int | None = None
    expected_behavior: str = ""
    source: str = "controlled synthetic validation"

    def resolved_expected_behavior(self) -> str:
        if self.expected_behavior:
            return self.expected_behavior
        try:
            return _resolve_category(self.expected_category).value
        except ValueError:
            return str(self.expected_category)

    def resolved_observed_circuit(self) -> Sequence[Any]:
        return self.expected_circuit if self.observed_circuit is None else self.observed_circuit


@dataclass
class ValidationRecord:
    case_id: str
    program: dict[str, list[Any]]
    expected_behavior: str
    expected_category: DiagnosisCategory | None
    observed_behavior: str
    observed_category: DiagnosisCategory | None
    diagnosis_result: DiagnosisResult | None
    relevant_candidates: list[dict[str, Any]]
    evidence: list[str]
    validation_satisfied: bool
    source: str
    shots: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        diagnosis = None
        if self.diagnosis_result is not None:
            stats = self.diagnosis_result.statistics_result
            diagnosis = {
                "anomaly_detected": self.diagnosis_result.anomaly_detected,
                "category": self.diagnosis_result.category.value,
                "summary": self.diagnosis_result.summary,
                "statistics": None if stats is None else {
                    "total_variation_distance": stats.total_variation_distance,
                    "threshold": stats.threshold,
                    "finite_shot_guard": stats.finite_shot_guard,
                    "shots": stats.shots,
                },
                "program_candidate_count": len(self.diagnosis_result.program_evidence_candidates),
                "hardware_candidate_count": len(self.diagnosis_result.hardware_evidence_candidates),
                "missing_evidence": list(self.diagnosis_result.missing_evidence),
            }

        return {
            "case_id": self.case_id,
            "program": self.program,
            "expected_behavior": self.expected_behavior,
            "expected_category": self.expected_category.value if self.expected_category else None,
            "observed_behavior": self.observed_behavior,
            "observed_category": self.observed_category.value if self.observed_category else None,
            "diagnosis": diagnosis,
            "relevant_candidates": self.relevant_candidates,
            "evidence": list(self.evidence),
            "validation_satisfied": self.validation_satisfied,
            "source": self.source,
            "shots": self.shots,
            "error": self.error,
        }


@dataclass
class ValidationReport:
    records: list[ValidationRecord] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(record.validation_satisfied for record in self.records)

    @property
    def failed(self) -> int:
        return len(self.records) - self.passed

    @property
    def all_passed(self) -> bool:
        return bool(self.records) and self.failed == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_count": len(self.records),
            "passed": self.passed,
            "failed": self.failed,
            "all_passed": self.all_passed,
            "records": [record.to_dict() for record in self.records],
        }


def _resolve_category(category: DiagnosisCategory | str) -> DiagnosisCategory:
    if isinstance(category, DiagnosisCategory):
        return category
    if not isinstance(category, str):
        raise ValueError("expected_category must be a DiagnosisCategory or category string")

    normalized = category.strip().upper()
    aliases = {
        "AMBIGUOUS": DiagnosisCategory.AMBIGUOUS,
        "INSUFFICIENT_EVIDENCE": DiagnosisCategory.INSUFFICIENT_EVIDENCE,
        "AMBIGUOUS / INSUFFICIENT_EVIDENCE": DiagnosisCategory.AMBIGUOUS,
    }
    if normalized in aliases:
        return aliases[normalized]
    try:
        return DiagnosisCategory[normalized]
    except KeyError as exc:
        raise ValueError(f"Unrecognized expected diagnosis category: {category!r}") from exc


def _candidate_records(result: DiagnosisResult) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for candidate in result.program_evidence_candidates:
        candidates.append({
            "source": "program",
            "component": candidate.operation,
            "index": candidate.index,
            "qubits": list(candidate.qubits),
            "score": candidate.score,
            "reason": candidate.reason,
            "evidence": list(candidate.evidence),
        })
    for candidate in result.hardware_evidence_candidates:
        component = list(candidate.component) if isinstance(candidate.component, tuple) else candidate.component
        candidates.append({
            "source": "hardware",
            "component_type": candidate.component_type,
            "component": component,
            "score": candidate.score,
            "reason": candidate.reason,
            "evidence": list(candidate.evidence),
        })
    return candidates


def _failed_record(case: Any, error: Exception, index: int | None = None) -> ValidationRecord:
    case_id = getattr(case, "case_id", None)
    if not isinstance(case_id, str) or not case_id:
        case_id = f"<invalid:{index}>" if index is not None else "<invalid>"
    expected_category = None
    expected_behavior = str(getattr(case, "expected_behavior", "") or getattr(case, "expected_category", ""))
    try:
        expected_category = _resolve_category(case.expected_category)
    except (AttributeError, ValueError):
        pass
    expected_circuit = getattr(case, "expected_circuit", ())
    observed_circuit = getattr(case, "observed_circuit", None)
    program = {
        "expected": list(expected_circuit) if isinstance(expected_circuit, Sequence) else [],
        "observed": list(observed_circuit) if isinstance(observed_circuit, Sequence) else [],
    }
    return ValidationRecord(
        case_id=case_id,
        program=program,
        expected_behavior=expected_behavior,
        expected_category=expected_category,
        observed_behavior="VALIDATION_ERROR",
        observed_category=None,
        diagnosis_result=None,
        relevant_candidates=[],
        evidence=[],
        validation_satisfied=False,
        source=str(getattr(case, "source", "controlled synthetic validation")),
        shots=getattr(case, "shots", None),
        error=f"{type(error).__name__}: {error}",
    )


def run_validation_case(case: ValidationCase) -> ValidationRecord:
    """Run one controlled case, returning a failed record for malformed input."""

    if not isinstance(case, ValidationCase):
        return _failed_record(case, TypeError("case must be a ValidationCase"))

    try:
        if not case.case_id or not isinstance(case.case_id, str):
            raise ValueError("case_id must be a non-empty string")
        expected_category = _resolve_category(case.expected_category)
        observed_circuit = case.resolved_observed_circuit()
        result = diagnose_execution(
            expected_distribution=case.expected_distribution,
            observed_distribution=case.observed_distribution,
            expected_circuit=case.expected_circuit,
            observed_circuit=observed_circuit,
            hardware_model=case.hardware_model,
            threshold=case.threshold,
            shots=case.shots,
        )
        observed_category = result.category
        candidates = _candidate_records(result)
        return ValidationRecord(
            case_id=case.case_id,
            program={
                "expected": list(case.expected_circuit),
                "observed": list(observed_circuit),
            },
            expected_behavior=case.resolved_expected_behavior(),
            expected_category=expected_category,
            observed_behavior=(
                f"{observed_category.value}; anomaly_detected={result.anomaly_detected}"
            ),
            observed_category=observed_category,
            diagnosis_result=result,
            relevant_candidates=candidates,
            evidence=list(result.evidence),
            validation_satisfied=observed_category is expected_category,
            source=case.source,
            shots=case.shots,
        )
    except Exception as exc:
        return _failed_record(case, exc)


def run_validation_suite(cases: Iterable[ValidationCase]) -> ValidationReport:
    """Run cases independently so a malformed scenario does not abort the suite."""

    if isinstance(cases, (str, bytes)):
        return ValidationReport(records=[_failed_record(cases, TypeError("cases must be an iterable of ValidationCase objects"))])
    try:
        iterator = iter(cases)
    except TypeError as exc:
        return ValidationReport(records=[_failed_record(cases, TypeError("cases must be iterable"))])

    records: list[ValidationRecord] = []
    seen_ids: set[str] = set()
    for index, case in enumerate(iterator):
        record = run_validation_case(case)
        if record.case_id in seen_ids:
            record.validation_satisfied = False
            record.error = f"Duplicate case identifier: {record.case_id!r}"
        seen_ids.add(record.case_id)
        records.append(record)
    return ValidationReport(records=records)


def standard_validation_cases() -> list[ValidationCase]:
    """Return deterministic, backend-independent synthetic pipeline scenarios."""

    no_state_change = {"0": 1.0, "1": 0.0}
    changed_state = {"0": 0.0, "1": 1.0}
    bell_expected = {"00": 0.0, "11": 1.0}
    bell_changed = {"00": 1.0, "11": 0.0}
    single_qubit_hardware = HardwareModel(1)
    single_qubit_hardware.set_qubit_properties(0, readout_error=0.4)

    ambiguous_hardware = HardwareModel(1)
    ambiguous_hardware.set_qubit_properties(0, readout_error=0.4)

    multi_hardware = HardwareModel(2, coupling_map=[(0, 1)])
    multi_hardware.set_qubit_properties(0, readout_error=0.4)
    multi_hardware.add_edge(0, 1, error=0.4)

    three_qubit_hardware = HardwareModel(3, coupling_map=[(0, 1), (1, 2)])
    three_qubit_hardware.add_edge(0, 1, error=0.4)

    low_shot_expected = {"0": 0.8, "1": 0.2}
    low_shot_observed = {"0": 0.72, "1": 0.28}

    return [
        ValidationCase(
            case_id="healthy_single_qubit",
            expected_category=DiagnosisCategory.NO_ANOMALY,
            expected_circuit=[("h", 0)],
            expected_distribution=no_state_change,
            observed_distribution=no_state_change,
            hardware_model=HardwareModel(1),
            shots=1000,
        ),
        ValidationCase(
            case_id="program_fault_single_qubit",
            expected_category=DiagnosisCategory.PROGRAM_FAULT,
            expected_circuit=[("x", 0)],
            observed_circuit=[("h", 0)],
            expected_distribution=no_state_change,
            observed_distribution=changed_state,
            hardware_model=HardwareModel(1),
            shots=1000,
        ),
        ValidationCase(
            case_id="hardware_fault_readout",
            expected_category=DiagnosisCategory.HARDWARE_ANOMALY,
            expected_circuit=[("measure", 0)],
            expected_distribution=no_state_change,
            observed_distribution=changed_state,
            hardware_model=single_qubit_hardware,
            shots=1000,
        ),
        ValidationCase(
            case_id="ambiguous_program_and_hardware",
            expected_category=DiagnosisCategory.AMBIGUOUS,
            expected_circuit=[("x", 0)],
            observed_circuit=[("h", 0)],
            expected_distribution=no_state_change,
            observed_distribution=changed_state,
            hardware_model=ambiguous_hardware,
            shots=1000,
        ),
        ValidationCase(
            case_id="insufficient_without_calibration",
            expected_category=DiagnosisCategory.INSUFFICIENT_EVIDENCE,
            expected_circuit=[("h", 0)],
            expected_distribution=no_state_change,
            observed_distribution=changed_state,
            shots=1000,
        ),
        ValidationCase(
            case_id="multiple_hardware_issues_two_qubits",
            expected_category=DiagnosisCategory.HARDWARE_ANOMALY,
            expected_circuit=[("x", 0), ("cnot", 0, 1), ("measure", 0)],
            expected_distribution=bell_expected,
            observed_distribution=bell_changed,
            hardware_model=multi_hardware,
            shots=1000,
        ),
        ValidationCase(
            case_id="three_qubit_operation",
            expected_category=DiagnosisCategory.HARDWARE_ANOMALY,
            expected_circuit=[("h", 0), ("ccx", 0, 1, 2)],
            expected_distribution={"000": 0.5, "111": 0.5},
            observed_distribution={"000": 1.0, "111": 0.0},
            hardware_model=three_qubit_hardware,
            shots=1000,
        ),
        ValidationCase(
            case_id="finite_shot_guard_low_shots",
            expected_category=DiagnosisCategory.NO_ANOMALY,
            expected_circuit=[("h", 0)],
            expected_distribution=low_shot_expected,
            observed_distribution=low_shot_observed,
            hardware_model=HardwareModel(1),
            shots=100,
        ),
        ValidationCase(
            case_id="finite_shot_guard_high_shots",
            expected_category=DiagnosisCategory.INSUFFICIENT_EVIDENCE,
            expected_circuit=[("h", 0)],
            expected_distribution=low_shot_expected,
            observed_distribution=low_shot_observed,
            hardware_model=HardwareModel(1),
            shots=1000,
        ),
        ValidationCase(
            case_id="partial_calibration_missing_gate_entry",
            expected_category=DiagnosisCategory.NO_ANOMALY,
            expected_circuit=[("rx", 0, 0.5)],
            expected_distribution=no_state_change,
            observed_distribution=no_state_change,
            hardware_model=HardwareModel(1),
            shots=1000,
        ),
    ]


__all__ = [
    "ValidationCase",
    "ValidationRecord",
    "ValidationReport",
    "run_validation_case",
    "run_validation_suite",
    "standard_validation_cases",
]
