"""Optional IBM Quantum validation adapter, isolated from the core simulator.

All IBM dependencies are imported lazily. Offline fixtures are explicitly
labelled and are never represented as results from IBM hardware.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from diagnosis_engine import DiagnosisResult, diagnose_execution
from hardware_model import HardwareModel
from quantum_simulator import QuantumSimulator
from statistics import compare_counts


REAL_EXECUTION_STATES = {"IMPLEMENTED", "OFFLINE_TESTED", "REAL_IBM_EXECUTED", "BLOCKED", "FAILED"}
SINGLE_QUBIT_METHODS = {
    "i": "id", "id": "id", "identity": "id", "x": "x", "y": "y", "z": "z", "h": "h",
    "s": "s", "sdg": "sdg", "t": "t", "tdg": "tdg", "sx": "sx", "sxdg": "sxdg",
}
ROTATION_METHODS = {"rx": "rx", "ry": "ry", "rz": "rz", "p": "p", "cp": "cp", "crx": "crx", "cry": "cry", "crz": "crz"}
TWO_QUBIT_METHODS = {"cx": "cx", "cnot": "cx", "cz": "cz", "swap": "swap", "ch": "ch"}
THREE_QUBIT_METHODS = {"ccx": "ccx", "toffoli": "ccx", "cswap": "cswap", "fredkin": "cswap"}
DEFAULT_MAX_CALIBRATION_AGE_SECONDS = 900.0


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _value_or_call(owner: Any, name: str, default: Any = None) -> Any:
    if owner is None:
        return default
    value = getattr(owner, name, default)
    if callable(value):
        try:
            return value()
        except Exception:
            return default
    return value


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    return str(value)


@dataclass
class IBMAccessStatus:
    status: str
    qiskit_available: bool
    runtime_available: bool
    credentials_detected: bool
    credential_source: str | None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "qiskit_available": self.qiskit_available,
            "runtime_available": self.runtime_available,
            "credentials_detected": self.credentials_detected,
            "credential_source": self.credential_source,
            "reason": self.reason,
        }


def check_ibm_access_status() -> IBMAccessStatus:
    """Inspect local dependency and credential availability without networking."""

    qiskit_available = importlib.util.find_spec("qiskit") is not None
    runtime_available = importlib.util.find_spec("qiskit_ibm_runtime") is not None
    token_variables = ("QISKIT_IBM_TOKEN", "IBM_QUANTUM_TOKEN", "QISKIT_IBM_API_KEY")
    token_variable = next((name for name in token_variables if os.environ.get(name)), None)
    instance_variable = next((name for name in ("QISKIT_IBM_INSTANCE", "IBM_QUANTUM_INSTANCE") if os.environ.get(name)), None)
    account_paths = (
        Path.home() / ".qiskit" / "qiskit-ibm.json",
        Path(os.environ.get("APPDATA", "")) / "Qiskit" / "qiskit-ibm.json" if os.environ.get("APPDATA") else None,
    )
    saved_account = next((path for path in account_paths if path is not None and path.is_file()), None)
    credentials_detected = bool(token_variable or saved_account)
    source = token_variable or ("saved Qiskit account configuration" if saved_account else None)

    if not qiskit_available or not runtime_available:
        missing = [name for name, available in (("qiskit", qiskit_available), ("qiskit-ibm-runtime", runtime_available)) if not available]
        return IBMAccessStatus("BLOCKED", qiskit_available, runtime_available, credentials_detected, source, f"Required package(s) not installed: {', '.join(missing)}")
    if not credentials_detected:
        return IBMAccessStatus("BLOCKED", True, True, False, None, "No IBM Quantum token or saved account configuration was detected.")
    _ = instance_variable
    return IBMAccessStatus("ACCESS_CONFIGURED", True, True, True, source)


@dataclass
class NormalizedOperation:
    name: str
    qubits: tuple[int, ...] = ()
    parameters: tuple[float, ...] = ()
    classical_bits: tuple[int, ...] = ()

    def to_tuple(self) -> tuple[Any, ...]:
        if self.name == "measure" and self.classical_bits:
            return (self.name, list(self.qubits), list(self.classical_bits))
        if self.name == "measure" and len(self.qubits) > 1:
            return (self.name, list(self.qubits))
        return (self.name, *self.qubits, *self.parameters)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "qubits": list(self.qubits),
            "parameters": list(self.parameters),
            "classical_bits": list(self.classical_bits),
        }


@dataclass
class CircuitPlan:
    num_qubits: int
    operations: list[NormalizedOperation]
    unsupported_operations: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def supported(self) -> bool:
        return not self.unsupported_operations and not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_qubits": self.num_qubits,
            "operations": [operation.to_dict() for operation in self.operations],
            "unsupported_operations": list(self.unsupported_operations),
            "errors": list(self.errors),
        }


def _parse_operation(item: Any) -> NormalizedOperation:
    if isinstance(item, Mapping):
        name = str(item.get("name", item.get("op", item.get("gate", "")))).strip().lower()
        qubits = item.get("qubits", [])
        parameters = item.get("parameters", item.get("params", []))
        classical_bits = item.get("classical_bits", item.get("clbits", []))
        if isinstance(qubits, int):
            qubits = [qubits]
        if isinstance(parameters, (int, float)):
            parameters = [parameters]
        if isinstance(classical_bits, int):
            classical_bits = [classical_bits]
        if not all(isinstance(value, (list, tuple)) for value in (qubits, parameters, classical_bits)):
            raise ValueError("operation qubits, parameters, and classical_bits must be sequences")
        return NormalizedOperation(
            name,
            tuple(_checked_index(value, "qubit") for value in qubits),
            tuple(float(value) for value in parameters),
            tuple(_checked_index(value, "classical bit") for value in classical_bits),
        )

    if not isinstance(item, (tuple, list)) or not item:
        raise ValueError(f"operation must be a non-empty tuple/list or mapping, got {item!r}")
    name = str(item[0]).strip().lower()
    operands = list(item[1:])
    if name == "measure":
        if not operands:
            return NormalizedOperation(name)
        if isinstance(operands[0], (list, tuple)):
            qubits = tuple(_checked_index(value, "qubit") for value in operands[0])
            classical_bits = tuple(_checked_index(value, "classical bit") for value in operands[1]) if len(operands) > 1 else ()
        else:
            qubits = (_checked_index(operands[0], "qubit"),)
            if len(operands) > 1:
                classical_operand = operands[1]
                if isinstance(classical_operand, (list, tuple)):
                    classical_bits = tuple(_checked_index(value, "classical bit") for value in classical_operand)
                else:
                    classical_bits = (_checked_index(classical_operand, "classical bit"),)
            else:
                classical_bits = ()
        if len(operands) > 2:
            raise ValueError("measure accepts qubits and optional classical bits")
        return NormalizedOperation(name, qubits, (), classical_bits)
    if name in {"reset", "barrier"}:
        if name == "reset" and len(operands) == 1 and isinstance(operands[0], (list, tuple)):
            qubits = tuple(_checked_index(value, "qubit") for value in operands[0])
        else:
            qubits = tuple(_checked_index(value, "qubit") for value in operands)
        return NormalizedOperation(name, qubits)

    gate_arity = {
        **{key: 1 for key in SINGLE_QUBIT_METHODS},
        **{key: 1 for key in {"rx", "ry", "rz", "p", "u1"}},
        "u2": 1, "u3": 1,
        **{key: 2 for key in TWO_QUBIT_METHODS},
        **{key: 2 for key in {"cp", "crx", "cry", "crz"}},
        **{key: 3 for key in THREE_QUBIT_METHODS},
    }
    parameter_arity = {"rx": 1, "ry": 1, "rz": 1, "p": 1, "u1": 1, "u2": 2, "u3": 3, "cp": 1, "crx": 1, "cry": 1, "crz": 1}
    if name not in gate_arity:
        return NormalizedOperation(name)
    qubit_count = gate_arity[name]
    parameter_count = parameter_arity.get(name, 0)
    if len(operands) != qubit_count + parameter_count:
        raise ValueError(f"{name} expects {qubit_count} qubits and {parameter_count} parameters, got {len(operands)} operands")
    qubits = tuple(_checked_index(value, "qubit") for value in operands[:qubit_count])
    try:
        parameters = tuple(float(value) for value in operands[qubit_count:])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} parameters must be numeric") from exc
    return NormalizedOperation(name, qubits, parameters)


def _checked_index(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} index must be a non-negative integer")
    return value


def build_circuit_plan(circuit: Sequence[Any], num_qubits: int | None = None) -> CircuitPlan:
    if not isinstance(circuit, Sequence) or isinstance(circuit, (str, bytes)):
        return CircuitPlan(num_qubits or 0, [], errors=["circuit must be a sequence of operations"])
    if num_qubits is not None and (not isinstance(num_qubits, int) or isinstance(num_qubits, bool) or num_qubits <= 0):
        return CircuitPlan(0, [], errors=["num_qubits must be a positive integer"])
    operations: list[NormalizedOperation] = []
    unsupported: list[dict[str, Any]] = []
    errors: list[str] = []
    maximum = -1
    for index, item in enumerate(circuit):
        try:
            operation = _parse_operation(item)
        except (TypeError, ValueError) as exc:
            errors.append(f"operation {index}: {exc}")
            continue
        supported_names = set(SINGLE_QUBIT_METHODS) | set(ROTATION_METHODS) | {"u1", "u2", "u3"} | set(TWO_QUBIT_METHODS) | set(THREE_QUBIT_METHODS) | {"measure", "reset", "barrier"}
        if operation.name not in supported_names:
            unsupported.append({"index": index, "operation": operation.name, "reason": "Operation is not supported by the adapter mapping."})
            continue
        if operation.name in {"u1", "u2", "u3"}:
            expected_parameters = {"u1": 1, "u2": 2, "u3": 3}[operation.name]
            if len(operation.parameters) != expected_parameters:
                errors.append(f"operation {index}: {operation.name} requires {expected_parameters} parameter(s)")
                continue
        if any(qubit >= (num_qubits if num_qubits is not None else 2**31) for qubit in operation.qubits):
            errors.append(f"operation {index}: qubit index exceeds configured circuit width")
            continue
        maximum = max(maximum, *operation.qubits) if operation.qubits else maximum
        operations.append(operation)

    inferred = maximum + 1
    width = num_qubits if num_qubits is not None else max(1, inferred)
    if not isinstance(width, int) or width <= 0:
        errors.append("num_qubits must be a positive integer")
        width = 0
    if num_qubits is not None and inferred > num_qubits:
        errors.append(f"circuit references {inferred} qubits but num_qubits={num_qubits}")
    return CircuitPlan(width, operations, unsupported, errors)


def _qiskit_circuit_from_plan(plan: CircuitPlan) -> tuple[Any | None, list[dict[str, Any]], str | None]:
    if not plan.supported:
        return None, [], "Circuit contains malformed or unsupported operations."
    try:
        from qiskit import QuantumCircuit
    except ImportError as exc:
        return None, [], f"Qiskit is not installed: {exc}"

    has_measurement = any(operation.name == "measure" for operation in plan.operations)
    circuit = QuantumCircuit(plan.num_qubits, plan.num_qubits if has_measurement else 0)
    mappings: list[dict[str, Any]] = []
    try:
        for operation in plan.operations:
            name, qubits, params = operation.name, operation.qubits, operation.parameters
            if name in SINGLE_QUBIT_METHODS:
                getattr(circuit, SINGLE_QUBIT_METHODS[name])(qubits[0])
            elif name in {"rx", "ry", "rz", "p", "cp", "crx", "cry", "crz"}:
                getattr(circuit, ROTATION_METHODS[name])(*params, *qubits)
            elif name == "u1":
                if hasattr(circuit, "u1"):
                    circuit.u1(params[0], qubits[0])
                else:
                    circuit.p(params[0], qubits[0])
                    mappings.append({"source": "u1", "target": "p", "reason": "Qiskit no longer exposes u1; p is the equivalent phase gate."})
            elif name == "u2":
                if hasattr(circuit, "u2"):
                    circuit.u2(*params, qubits[0])
                else:
                    circuit.u(np.pi / 2, *params, qubits[0])
                    mappings.append({"source": "u2", "target": "u", "reason": "Qiskit no longer exposes u2; mapped explicitly to u(pi/2, phi, lambda)."})
            elif name == "u3":
                if hasattr(circuit, "u3"):
                    circuit.u3(*params, qubits[0])
                else:
                    circuit.u(*params, qubits[0])
                    mappings.append({"source": "u3", "target": "u", "reason": "Qiskit no longer exposes u3; mapped explicitly to u(theta, phi, lambda)."})
            elif name in TWO_QUBIT_METHODS:
                getattr(circuit, TWO_QUBIT_METHODS[name])(*qubits)
            elif name in THREE_QUBIT_METHODS:
                getattr(circuit, THREE_QUBIT_METHODS[name])(*qubits)
            elif name == "measure":
                if not qubits:
                    circuit.measure_all()
                elif operation.classical_bits:
                    if len(qubits) != len(operation.classical_bits):
                        raise ValueError("measurement qubits and classical bits must have equal lengths")
                    circuit.measure(list(qubits), list(operation.classical_bits))
                else:
                    circuit.measure(list(qubits), list(qubits))
            elif name == "reset":
                for qubit in qubits:
                    circuit.reset(qubit)
            elif name == "barrier":
                circuit.barrier(*qubits)
    except Exception as exc:
        return None, mappings, f"Qiskit circuit construction failed: {type(exc).__name__}: {exc}"
    return circuit, mappings, None


@dataclass
class CircuitConversionResult:
    status: str
    circuit: Any | None
    plan: CircuitPlan
    gate_mappings: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


def convert_to_qiskit_circuit(circuit: Sequence[Any], num_qubits: int | None = None) -> CircuitConversionResult:
    plan = build_circuit_plan(circuit, num_qubits)
    if not plan.supported:
        status = "UNSUPPORTED" if plan.unsupported_operations else "FAILED"
        return CircuitConversionResult(status, None, plan, error="; ".join(plan.errors) if plan.errors else "Unsupported circuit operations are present.")
    qiskit_circuit, mappings, error = _qiskit_circuit_from_plan(plan)
    if error:
        return CircuitConversionResult("BLOCKED" if "not installed" in error.lower() else "FAILED", None, plan, mappings, error)
    return CircuitConversionResult("CONVERTED", qiskit_circuit, plan, mappings)


def counts_to_probabilities(counts: Mapping[str, int], *, shots: int | None = None) -> dict[str, float]:
    if not isinstance(counts, Mapping) or not counts:
        raise ValueError("counts must be a non-empty mapping")
    normalized: dict[str, int] = {}
    for outcome, count in counts.items():
        if not isinstance(outcome, str):
            raise TypeError("count outcome keys must be strings")
        if not isinstance(count, (int, np.integer)) or isinstance(count, bool) or count < 0:
            raise ValueError(f"count for {outcome!r} must be a non-negative integer")
        compact_outcome = "".join(outcome.split())
        if not compact_outcome or any(bit not in "01" for bit in compact_outcome):
            raise ValueError(f"count outcome {outcome!r} must contain only binary measurement bits")
        key = compact_outcome[::-1]
        normalized[key] = normalized.get(key, 0) + int(count)
    widths = {len(outcome) for outcome in normalized}
    if len(widths) != 1:
        raise ValueError("count outcomes must have a consistent measurement width")
    total = sum(normalized.values())
    if total <= 0:
        raise ValueError("counts must have a positive total")
    if shots is not None:
        if not isinstance(shots, int) or isinstance(shots, bool) or shots <= 0:
            raise ValueError("shots must be a positive integer")
        if total != shots:
            raise ValueError(f"count total {total} does not match requested shots {shots}")
    return {outcome: count / total for outcome, count in sorted(normalized.items())}


def _validate_execution_interval(start: str | None, end: str | None) -> None:
    """Require a complete, timezone-aware, non-reversed execution interval."""

    if not start or not end:
        raise ValueError("execution start and end timestamps are required before diagnosis")
    try:
        start_at = datetime.fromisoformat(start.replace("Z", "+00:00"))
        end_at = datetime.fromisoformat(end.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("execution timestamps must be valid ISO-8601 values") from exc
    if start_at.tzinfo is None or end_at.tzinfo is None:
        raise ValueError("execution timestamps must include a timezone")
    if end_at.astimezone(timezone.utc) < start_at.astimezone(timezone.utc):
        raise ValueError("execution end timestamp precedes execution start timestamp")


def _parameter_map(parameters: Any) -> dict[str, Any]:
    if parameters is None:
        return {}
    result: dict[str, Any] = {}
    for parameter in parameters:
        if isinstance(parameter, Mapping):
            name, value = parameter.get("name"), parameter.get("value")
            unit = parameter.get("unit")
        else:
            name, value, unit = getattr(parameter, "name", None), getattr(parameter, "value", None), getattr(parameter, "unit", None)
        if name is not None:
            result[str(name).lower()] = {"value": _json_safe(value), "unit": _json_safe(unit)}
    return result


def _property_values(properties: Any, qubit: int) -> dict[str, Any]:
    values: dict[str, Any] = {}
    qubit_rows = getattr(properties, "qubits", None)
    if isinstance(qubit_rows, (list, tuple)) and qubit < len(qubit_rows):
        for parameter in qubit_rows[qubit] or []:
            entry = _parameter_map([parameter])
            for name, value in entry.items():
                values[name] = value
    getter = getattr(properties, "qubit_property", None)
    if callable(getter):
        for name in ("T1", "T2", "readout_error"):
            try:
                value = getter(qubit, name)
                if value is not None and name.lower() not in values:
                    values[name.lower()] = {"value": _json_safe(value[0] if isinstance(value, (tuple, list)) else value), "unit": _json_safe(value[1] if isinstance(value, (tuple, list)) and len(value) > 1 else None)}
            except Exception:
                continue
    return values


def _numeric_property(values: Mapping[str, Any], names: Sequence[str]) -> float | None:
    for name in names:
        entry = values.get(name.lower())
        if isinstance(entry, Mapping):
            entry = entry.get("value")
        try:
            if entry is not None:
                return float(entry)
        except (TypeError, ValueError):
            continue
    return None


def _t_value_in_microseconds(value: float | None, unit: Any) -> float | None:
    if value is None or unit is None:
        return value
    normalized = str(unit).strip().lower().replace("μ", "u").replace("µ", "u")
    if normalized in {"s", "sec", "second", "seconds"}:
        return value * 1_000_000
    if normalized in {"ms", "millisecond", "milliseconds"}:
        return value * 1_000
    if normalized in {"ns", "nanosecond", "nanoseconds"}:
        return value / 1_000
    return value


@dataclass
class IBMCalibrationCapture:
    backend_name: str | None
    timestamp: str | None
    n_qubits: int | None
    coupling_map: list[tuple[int, int]] | None
    backend_status: dict[str, Any] | None
    qubits: dict[int, dict[str, Any]]
    gates: list[dict[str, Any]]
    unavailable_fields: list[str]
    hardware_model: HardwareModel | None
    retrieval_status: str
    capture_source: str = "IBM_BACKEND_API"
    errors: list[str] = field(default_factory=list)
    retrieved_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "timestamp": self.timestamp,
            "n_qubits": self.n_qubits,
            "coupling_map": _json_safe(self.coupling_map),
            "backend_status": _json_safe(self.backend_status),
            "qubits": _json_safe(self.qubits),
            "gates": _json_safe(self.gates),
            "unavailable_fields": list(self.unavailable_fields),
            "retrieval_status": self.retrieval_status,
            "capture_source": self.capture_source,
            "retrieved_at": self.retrieved_at,
            "errors": list(self.errors),
        }


def capture_backend_calibration(backend: Any, *, capture_source: str = "IBM_BACKEND_API") -> IBMCalibrationCapture:
    """Extract available properties from a Qiskit-like backend object."""

    retrieved_at = utc_timestamp()
    errors: list[str] = []
    unavailable: list[str] = []
    name = _value_or_call(backend, "name")
    if name is None:
        name = getattr(backend, "backend_name", None)
    configuration = _value_or_call(backend, "configuration")
    n_qubits = getattr(backend, "num_qubits", None)
    if n_qubits is None:
        n_qubits = getattr(configuration, "n_qubits", None)
    try:
        n_qubits = int(n_qubits) if n_qubits is not None else None
    except (TypeError, ValueError):
        n_qubits = None
    if n_qubits is not None and n_qubits <= 0:
        errors.append(f"Malformed backend qubit count: {n_qubits}")
        n_qubits = None
    if n_qubits is None:
        unavailable.append("n_qubits")

    raw_coupling = getattr(backend, "coupling_map", None)
    if raw_coupling is None:
        raw_coupling = getattr(configuration, "coupling_map", None)
    if hasattr(raw_coupling, "get_edges"):
        try:
            raw_coupling = raw_coupling.get_edges()
        except Exception as exc:
            errors.append(f"Coupling map retrieval failed: {exc}")
            raw_coupling = None
    try:
        coupling_map = (
            [tuple(_checked_index(index, "coupling qubit") for index in edge) for edge in raw_coupling]
            if raw_coupling is not None else None
        )
        if coupling_map is not None and any(len(edge) != 2 for edge in coupling_map):
            raise ValueError("coupling edges must have exactly two qubit indices")
        if coupling_map is not None:
            for left, right in coupling_map:
                if left < 0 or right < 0 or left == right:
                    raise ValueError(f"invalid coupling edge ({left}, {right})")
                if n_qubits is not None and (left >= n_qubits or right >= n_qubits):
                    raise ValueError(f"coupling edge ({left}, {right}) exceeds backend qubit count {n_qubits}")
    except (TypeError, ValueError) as exc:
        errors.append(f"Malformed coupling map: {exc}")
        coupling_map = None
    if coupling_map is None:
        unavailable.append("coupling_map")

    status_object = _value_or_call(backend, "status")
    backend_status = None
    if status_object is not None:
        backend_status = {
            "operational": _json_safe(getattr(status_object, "operational", None)),
            "status_msg": _json_safe(getattr(status_object, "status_msg", None)),
            "pending_jobs": _json_safe(getattr(status_object, "pending_jobs", None)),
        }
    else:
        unavailable.append("backend_status")

    try:
        properties_value = getattr(backend, "properties", None)
        properties = properties_value() if callable(properties_value) else properties_value
    except Exception as exc:
        properties = None
        errors.append(f"Backend properties retrieval failed: {exc}")
    calibration_timestamp = getattr(properties, "last_update_date", None)
    if hasattr(calibration_timestamp, "isoformat"):
        calibration_timestamp = calibration_timestamp.isoformat()
    elif calibration_timestamp is not None:
        calibration_timestamp = str(calibration_timestamp)
    else:
        unavailable.append("calibration_timestamp")

    try:
        model = HardwareModel(n_qubits, coupling_map=coupling_map, backend_name=str(name or "unknown_backend")) if n_qubits is not None else None
    except (TypeError, ValueError) as exc:
        errors.append(f"HardwareModel conversion failed: {exc}")
        model = HardwareModel(n_qubits, backend_name=str(name or "unknown_backend")) if n_qubits is not None else None
    qubits: dict[int, dict[str, Any]] = {}
    if n_qubits is not None:
        for qubit in range(n_qubits):
            values = _property_values(properties, qubit)
            t1 = _numeric_property(values, ("t1",))
            t2 = _numeric_property(values, ("t2",))
            readout = _numeric_property(values, ("readout_error", "readouterror"))
            qubits[qubit] = {
                "t1": t1,
                "t1_unit": values.get("t1", {}).get("unit") if isinstance(values.get("t1"), Mapping) else None,
                "t2": t2,
                "t2_unit": values.get("t2", {}).get("unit") if isinstance(values.get("t2"), Mapping) else None,
                "readout_error": readout,
            }
            for metric, value in (("t1", t1), ("t2", t2), ("readout_error", readout)):
                if value is None:
                    unavailable.append(f"qubits[{qubit}].{metric}")
            if model is not None:
                t1_unit = values.get("t1", {}).get("unit") if isinstance(values.get("t1"), Mapping) else None
                t2_unit = values.get("t2", {}).get("unit") if isinstance(values.get("t2"), Mapping) else None
                supplied = {
                    key: value for key, value in (
                        ("t1", _t_value_in_microseconds(t1, t1_unit)),
                        ("t2", _t_value_in_microseconds(t2, t2_unit)),
                        ("readout_error", readout),
                    ) if value is not None
                }
                try:
                    model.set_qubit_properties(qubit, **supplied)
                except (TypeError, ValueError) as exc:
                    errors.append(f"Qubit {qubit} calibration rejected: {exc}")

    gates: list[dict[str, Any]] = []
    properties_gates = getattr(properties, "gates", None) or []
    if properties is None:
        unavailable.append("gate_calibrations")
    for gate in properties_gates:
        gate_name = getattr(gate, "gate", getattr(gate, "name", None))
        gate_qubits = getattr(gate, "qubits", None)
        if gate_name is None or gate_qubits is None:
            errors.append("Malformed gate calibration entry missing gate name or qubits")
            continue
        try:
            normalized_gate_qubits = [_checked_index(value, "gate calibration qubit") for value in gate_qubits]
            if n_qubits is not None and any(qubit >= n_qubits for qubit in normalized_gate_qubits):
                raise ValueError("gate calibration qubit exceeds backend width")
        except (TypeError, ValueError) as exc:
            errors.append(f"Malformed gate calibration entry {gate_name!r}: {exc}")
            continue
        params = _parameter_map(getattr(gate, "parameters", None))
        error = _numeric_property(params, ("gate_error", "error"))
        duration = _numeric_property(params, ("gate_length", "duration"))
        entry = {
            "gate": str(gate_name).lower(),
            "qubits": normalized_gate_qubits,
            "error": error,
            "error_unit": params.get("gate_error", {}).get("unit") if isinstance(params.get("gate_error"), Mapping) else None,
            "duration": duration,
            "duration_unit": params.get("gate_length", {}).get("unit") if isinstance(params.get("gate_length"), Mapping) else None,
            "supported": True,
        }
        gates.append(entry)
        for metric in ("error", "duration"):
            if entry[metric] is None:
                unavailable.append(f"gates[{gate_name},{entry['qubits']}].{metric}")
        if model is not None:
            supplied = {key: entry[key] for key in ("error", "duration") if entry[key] is not None}
            try:
                model.add_gate_calibration(str(gate_name), qubits=normalized_gate_qubits, **supplied)
            except (TypeError, ValueError) as exc:
                errors.append(f"Gate calibration {gate_name} {gate_qubits} rejected: {exc}")

    if not properties_gates:
        unavailable.append("gate_calibrations")
    if model is not None:
        model.calibration_timestamp = calibration_timestamp
        model.calibration_timestamp_provenance = "reported" if calibration_timestamp else "unknown"
        model.backend_name = str(name or "unknown_backend")

    retrieval_status = "AVAILABLE" if properties is not None and n_qubits is not None else "PARTIAL"
    return IBMCalibrationCapture(
        backend_name=str(name) if name is not None else None,
        timestamp=calibration_timestamp,
        n_qubits=n_qubits,
        coupling_map=coupling_map,
        backend_status=backend_status,
        qubits=qubits,
        gates=gates,
        unavailable_fields=sorted(set(unavailable)),
        hardware_model=model,
        retrieval_status=retrieval_status,
        capture_source=capture_source,
        retrieved_at=retrieved_at,
        errors=errors,
    )


def calibration_freshness(
    capture: IBMCalibrationCapture,
    *,
    max_age_seconds: float = DEFAULT_MAX_CALIBRATION_AGE_SECONDS,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Assess calibration age without treating a missing timestamp as current."""

    if isinstance(max_age_seconds, bool) or not isinstance(max_age_seconds, (int, float)) or not math.isfinite(float(max_age_seconds)) or max_age_seconds <= 0:
        raise ValueError("max_age_seconds must be a positive number")
    stamp = capture.timestamp
    if not stamp:
        return {"status": "MISSING_TIMESTAMP", "age_seconds": None, "max_age_seconds": float(max_age_seconds)}
    try:
        parsed = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return {"status": "INVALID_TIMESTAMP", "age_seconds": None, "max_age_seconds": float(max_age_seconds)}
    if parsed.tzinfo is None:
        return {"status": "INVALID_TIMESTAMP", "age_seconds": None, "max_age_seconds": float(max_age_seconds)}
    reference = now or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        raise ValueError("now must include a timezone")
    age = (reference - parsed.astimezone(timezone.utc)).total_seconds()
    if age < 0:
        return {"status": "FUTURE_TIMESTAMP", "age_seconds": age, "max_age_seconds": float(max_age_seconds)}
    return {
        "status": "FRESH" if age <= max_age_seconds else "STALE",
        "age_seconds": age,
        "max_age_seconds": float(max_age_seconds),
    }


def _calibration_snapshot_id(capture: IBMCalibrationCapture | None) -> str | None:
    if capture is None:
        return None
    payload = json.dumps(capture.to_dict(), sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def preflight_ibm_backend(
    backend_name: str,
    *,
    token: str | None = None,
    instance: str | None = None,
    channel: str = "ibm_quantum_platform",
    service: Any | None = None,
    max_calibration_age_seconds: float = DEFAULT_MAX_CALIBRATION_AGE_SECONDS,
    required_fields: Sequence[str] = (),
    now: datetime | None = None,
) -> dict[str, Any]:
    """Perform a read-only backend/calibration check; this function submits no circuit."""

    if not isinstance(backend_name, str) or not backend_name.strip():
        raise ValueError("backend_name must be a non-empty string")
    if isinstance(max_calibration_age_seconds, bool) or not isinstance(max_calibration_age_seconds, (int, float)) or not math.isfinite(float(max_calibration_age_seconds)) or max_calibration_age_seconds <= 0:
        raise ValueError("max_calibration_age_seconds must be a positive number")
    if isinstance(required_fields, str) or any(not isinstance(name, str) for name in required_fields):
        raise ValueError("required_fields must be a sequence of field-name strings")
    fixture_service = service is not None
    access = (
        IBMAccessStatus("SERVICE_PROVIDED", True, True, False, "injected service fixture")
        if fixture_service
        else _explicit_credential_access_status(token)
    )
    checks: list[dict[str, str]] = []
    if fixture_service:
        checks.append({"name": "dependencies", "status": "NOT_REQUIRED_FOR_INJECTED_SERVICE", "detail": "Caller supplied the service object; this does not verify installed IBM dependencies."})
        checks.append({"name": "credentials", "status": "NOT_VERIFIED", "detail": "Caller-provided services are not evidence of IBM authentication."})
    else:
        dependencies_ok = access.qiskit_available and access.runtime_available
        checks.append({"name": "dependencies", "status": "PASS" if dependencies_ok else "FAIL", "detail": "Qiskit and qiskit-ibm-runtime availability checked locally."})
        credentials_ok = isinstance(token, str) and bool(token.strip())
        checks.append({"name": "explicit_credentials", "status": "PASS" if credentials_ok else "FAIL", "detail": "An explicit token was supplied." if credentials_ok else "No explicit token was supplied."})
    if not fixture_service and (not access.qiskit_available or not access.runtime_available):
        return {"status": "BLOCKED", "blocker": "DEPENDENCIES_MISSING", "backend": backend_name, "access": access.to_dict(), "checks": checks, "calibration": None, "calibration_assessment": None, "job_submitted": False, "transpilation_performed": False, "error": access.reason}
    if not fixture_service and (not isinstance(token, str) or not token.strip()):
        return {"status": "BLOCKED", "blocker": "CREDENTIALS_MISSING", "backend": backend_name, "access": access.to_dict(), "checks": checks, "calibration": None, "calibration_assessment": None, "job_submitted": False, "transpilation_performed": False, "error": access.reason}
    try:
        runtime_service = service if fixture_service else _runtime_service(token, instance, channel)
        backend = runtime_service.backend(backend_name)
    except Exception as exc:
        checks.append({"name": "backend_access", "status": "FAIL", "detail": f"{type(exc).__name__}: {exc}"})
        return {"status": "BLOCKED", "blocker": "BACKEND_ACCESS_FAILED", "backend": backend_name, "access": access.to_dict(), "checks": checks, "calibration": None, "calibration_assessment": None, "job_submitted": False, "transpilation_performed": False, "error": f"{type(exc).__name__}: {exc}"}

    capture = capture_backend_calibration(backend, capture_source="INJECTED_SERVICE_FIXTURE" if fixture_service else "IBM_BACKEND_API")
    freshness = calibration_freshness(capture, max_age_seconds=max_calibration_age_seconds, now=now)
    status_value = (capture.backend_status or {}).get("operational")
    missing_required = sorted(set(required_fields).intersection(capture.unavailable_fields))
    completeness_gaps = missing_required if required_fields else list(capture.unavailable_fields)
    identity_ok = capture.backend_name == backend_name
    checks.extend([
        {"name": "backend_identity", "status": "PASS" if identity_ok else "FAIL", "detail": f"Requested {backend_name!r}; received {capture.backend_name!r}."},
        {"name": "backend_status", "status": "PASS" if status_value is True else ("FAIL" if status_value is False else "UNVERIFIED"), "detail": "Backend reports operational." if status_value is True else ("Backend reports non-operational." if status_value is False else "Operational status was unavailable." )},
        {"name": "calibration_timestamp", "status": "PASS" if freshness["status"] == "FRESH" else "FAIL", "detail": f"Freshness result: {freshness['status']} (age={freshness['age_seconds']}, limit={freshness['max_age_seconds']} seconds)."},
        {"name": "calibration_completeness", "status": "PASS" if not completeness_gaps and not capture.errors else "FAIL", "detail": "Required calibration fields are available." if not completeness_gaps and not capture.errors else "Unavailable required fields or calibration parse errors are present."},
        {"name": "qubit_and_coupling_indices", "status": "PASS" if capture.n_qubits and capture.coupling_map is not None and not capture.errors else "FAIL", "detail": "Qubit count and coupling indices parsed without errors." if capture.n_qubits and capture.coupling_map is not None and not capture.errors else "Qubit count, coupling map, or calibration index validation is incomplete or invalid."},
    ])
    if status_value is False:
        blocker = "BACKEND_UNAVAILABLE"
    elif not identity_ok:
        blocker = "BACKEND_IDENTITY_MISMATCH"
    elif freshness["status"] != "FRESH":
        blocker = "CALIBRATION_" + freshness["status"]
    elif status_value is not True:
        blocker = "BACKEND_STATUS_UNKNOWN"
    elif completeness_gaps:
        blocker = "INCOMPLETE_CALIBRATION"
    elif capture.errors:
        blocker = "CALIBRATION_INVALID"
    elif capture.n_qubits is None or capture.coupling_map is None:
        blocker = "TOPOLOGY_INCOMPLETE"
    else:
        blocker = None
    live_backend = not fixture_service
    status = "PREFLIGHT_READY" if blocker is None and live_backend else ("PREFLIGHT_INJECTED_READY" if blocker is None else "BLOCKED")
    snapshot_id = _calibration_snapshot_id(capture)
    return {
        "status": status,
        "blocker": blocker,
        "backend": capture.backend_name,
        "requested_backend": backend_name,
        "access": access.to_dict(),
        "source": "CALLER_PROVIDED_SERVICE" if fixture_service else "IBM_BACKEND_API",
        "evidence_class": "INJECTED_OR_MOCKED" if fixture_service else ("AUTHENTICATED_BACKEND_API" if blocker is None else "LIVE_ACCESS_ATTEMPT"),
        "real_backend_contact": None if fixture_service else True,
        "checks": checks,
        "job_submitted": False,
        "transpilation_performed": False,
        "calibration": capture.to_dict(),
        "calibration_snapshot_id": snapshot_id,
        "calibration_assessment": {**freshness, "backend_name": capture.backend_name, "calibration_timestamp": capture.timestamp, "calibration_retrieved_at": capture.retrieved_at, "calibration_capture_source": capture.capture_source, "calibration_snapshot_id": snapshot_id, "completeness": "COMPLETE_FOR_REQUEST" if not completeness_gaps and not capture.errors else "INCOMPLETE", "missing_required_fields": missing_required, "unavailable_fields": capture.unavailable_fields, "errors": capture.errors},
        "error": None if blocker is None else blocker,
    }


def inspect_ibm_backend_calibration(
    backend_name: str,
    *,
    service: Any | None = None,
    channel: str = "ibm_quantum_platform",
) -> IBMCalibrationCapture:
    """Retrieve current backend calibration metadata without submitting a job."""
    runtime_service = service if service is not None else _environment_runtime_service(channel)
    backend = runtime_service.backend(backend_name)
    return capture_backend_calibration(backend)


def hardware_model_from_ibm_calibration(capture: IBMCalibrationCapture) -> HardwareModel:
    """Map an IBM calibration capture to the simulator's hardware model."""
    if not isinstance(capture, IBMCalibrationCapture):
        raise TypeError("capture must be an IBMCalibrationCapture")
    if capture.n_qubits is None:
        raise ValueError("IBM calibration does not provide the backend qubit count")
    if capture.backend_name is None:
        raise ValueError("IBM calibration does not provide the backend name")

    model = HardwareModel(
        capture.n_qubits,
        coupling_map=capture.coupling_map,
        backend_name=capture.backend_name,
        calibration_timestamp=capture.timestamp,
    )
    # HardwareModel stores undirected edges for routing and a separate
    # directional adjacency map. Preserve the backend's directed topology in
    # that existing adjacency representation.
    if capture.coupling_map is not None:
        model._directional_connectivity = {qubit: [] for qubit in range(capture.n_qubits)}
        for left, right in capture.coupling_map:
            if right not in model._directional_connectivity[left]:
                model._directional_connectivity[left].append(right)

    for qubit, calibration in capture.qubits.items():
        properties = model.get_qubit_properties(int(qubit))
        properties.t1 = _t_value_in_microseconds(calibration["t1"], calibration.get("t1_unit"))
        properties.t2 = _t_value_in_microseconds(calibration["t2"], calibration.get("t2_unit"))
        properties.readout_error = calibration["readout_error"]

    for calibration in capture.gates:
        model.add_gate_calibration(
            calibration["gate"],
            qubits=calibration["qubits"],
            error=calibration["error"],
            duration=calibration["duration"],
            supported=calibration["supported"],
            error_unit=calibration.get("error_unit"),
            duration_unit=calibration.get("duration_unit"),
        )
    # The HardwareModel constructor uses a local timestamp when passed None;
    # retain the source value even when the IBM snapshot lacks one.
    model.calibration_timestamp = capture.timestamp
    model.calibration_timestamp_provenance = "reported" if capture.timestamp else "unknown"
    return model


def _runtime_service(token: str, instance: str | None = None, channel: str = "ibm_quantum_platform") -> Any:
    from qiskit_ibm_runtime import QiskitRuntimeService

    kwargs: dict[str, Any] = {"channel": channel, "token": token}
    if instance:
        kwargs["instance"] = instance
    return QiskitRuntimeService(**kwargs)


def _explicit_credential_access_status(token: str | None) -> IBMAccessStatus:
    qiskit_available = importlib.util.find_spec("qiskit") is not None
    runtime_available = importlib.util.find_spec("qiskit_ibm_runtime") is not None
    if not qiskit_available or not runtime_available:
        missing = [name for name, available in (("qiskit", qiskit_available), ("qiskit-ibm-runtime", runtime_available)) if not available]
        return IBMAccessStatus("BLOCKED", qiskit_available, runtime_available, bool(token), "explicit token" if token else None, f"Required package(s) not installed: {', '.join(missing)}")
    if not isinstance(token, str) or not token.strip():
        return IBMAccessStatus("BLOCKED", True, True, False, None, "An explicit IBM Quantum token is required for live IBM operations.")
    return IBMAccessStatus("ACCESS_CONFIGURED", True, True, True, "explicit token")


def _environment_runtime_service(channel: str) -> Any:
    """Legacy CLI-only service creation using the user's configured environment."""
    token = next((os.environ[name] for name in ("QISKIT_IBM_TOKEN", "IBM_QUANTUM_TOKEN", "QISKIT_IBM_API_KEY") if os.environ.get(name)), None)
    instance = next((os.environ[name] for name in ("QISKIT_IBM_INSTANCE", "IBM_QUANTUM_INSTANCE") if os.environ.get(name)), None)
    if token:
        return _runtime_service(token, instance, channel)
    from qiskit_ibm_runtime import QiskitRuntimeService
    return QiskitRuntimeService(channel=channel)


def discover_ibm_backends(service: Any | None = None, *, token: str | None = None, instance: str | None = None, channel: str = "ibm_quantum_platform") -> dict[str, Any]:
    if service is None:
        access = _explicit_credential_access_status(token)
    else:
        access = IBMAccessStatus("SERVICE_PROVIDED", True, True, False, "provided Runtime service")
    if service is None and (not isinstance(token, str) or not token.strip()):
        return {"status": "BLOCKED", "discovery_source": "IBM_RUNTIME", "backends": [], "access": access.to_dict(), "error": access.reason}
    if service is None and access.status != "ACCESS_CONFIGURED":
        return {"status": "BLOCKED", "discovery_source": "IBM_RUNTIME", "backends": [], "access": access.to_dict(), "error": access.reason}
    try:
        runtime_service = service if service is not None else _runtime_service(token, instance, channel)
        backends = runtime_service.backends()
        found = []
        for backend in backends:
            name = _value_or_call(backend, "name") or getattr(backend, "backend_name", None)
            status = _value_or_call(backend, "status")
            found.append({
                "name": str(name) if name is not None else None,
                "num_qubits": getattr(backend, "num_qubits", None),
                "operational": getattr(status, "operational", None) if status is not None else None,
            })
        return {"status": "PARSED_INJECTED_SERVICE" if service is not None else "DISCOVERED", "discovery_source": "INJECTED_SERVICE_FIXTURE" if service is not None else "IBM_RUNTIME", "backends": found, "access": access.to_dict(), "error": None}
    except Exception as exc:
        return {"status": "FAILED", "discovery_source": "INJECTED_SERVICE_FIXTURE" if service is not None else "IBM_RUNTIME", "backends": [], "access": access.to_dict(), "error": f"{type(exc).__name__}: {exc}"}


def ideal_distribution(circuit: Sequence[Any], num_qubits: int, *, seed: int = 1234) -> dict[str, float]:
    plan = build_circuit_plan(circuit, num_qubits)
    if not plan.supported:
        reasons = [*plan.errors, *(item["reason"] for item in plan.unsupported_operations)]
        raise ValueError("Cannot simulate circuit: " + "; ".join(reasons))
    simulator_ops = []
    for operation in plan.operations:
        if operation.name == "measure":
            continue
        if operation.name == "reset":
            simulator_ops.extend(("reset", qubit) for qubit in operation.qubits)
        else:
            simulator_ops.append(operation.to_tuple())
    simulator = QuantumSimulator(num_qubits, seed=seed)
    simulator.run_circuit(simulator_ops)
    probabilities = simulator.get_probabilities()
    return {f"{index:0{num_qubits}b}": float(probability) for index, probability in enumerate(probabilities)}


def _deterministic_localization_circuit(plan: CircuitPlan) -> list[tuple[Any, ...]]:
    """Replace stochastic measurements with no-op barriers only for localizers."""

    operations: list[tuple[Any, ...]] = []
    for operation in plan.operations:
        if operation.name == "measure":
            qubits = operation.qubits or tuple(range(plan.num_qubits))
            operations.append(("barrier", *qubits))
        else:
            operations.append(operation.to_tuple())
    return operations


def _counts_from_probabilities(distribution: Mapping[str, float], shots: int) -> dict[str, int]:
    outcomes = sorted(distribution)
    exact = [max(0.0, float(distribution[outcome])) * shots for outcome in outcomes]
    counts = [int(value) for value in exact]
    remainder = shots - sum(counts)
    order = sorted(range(len(outcomes)), key=lambda index: exact[index] - counts[index], reverse=True)
    for index in order[:remainder]:
        counts[index] += 1
    return {outcome: count for outcome, count in zip(outcomes, counts) if count > 0}


def _candidate_payload(diagnosis: DiagnosisResult | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if diagnosis is None:
        return [], []
    program = [
        {"index": candidate.index, "operation": candidate.operation, "qubits": list(candidate.qubits), "score": candidate.score, "reason": candidate.reason, "evidence": list(candidate.evidence)}
        for candidate in diagnosis.program_evidence_candidates
    ]
    hardware = []
    for candidate in diagnosis.hardware_evidence_candidates:
        component = list(candidate.component) if isinstance(candidate.component, tuple) else candidate.component
        hardware.append({"component_type": candidate.component_type, "component": component, "score": candidate.score, "reason": candidate.reason, "evidence": list(candidate.evidence)})
    return program, hardware


@dataclass
class IBMValidationResult:
    overall_status: str
    execution_status: str
    execution_origin: str
    backend_name: str | None
    job_id: str | None
    shots: int
    execution_timestamp: str
    ideal_distribution: dict[str, float] | None
    observed_counts: dict[str, int] | None
    observed_distribution: dict[str, float] | None
    anomaly: dict[str, Any] | None
    diagnosis: dict[str, Any] | None
    program_candidates: list[dict[str, Any]]
    hardware_candidates: list[dict[str, Any]]
    calibration_snapshot: dict[str, Any] | None
    transpilation_info: dict[str, Any]
    conversion_info: dict[str, Any]
    access: dict[str, Any]
    errors: list[str] = field(default_factory=list)
    real_hardware_executed: bool = False
    execution_start_timestamp: str | None = None
    execution_end_timestamp: str | None = None
    calibration_snapshot_id: str | None = None
    calibration_assessment: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "implementation_status": "IMPLEMENTED",
            "offline_status": "OFFLINE_TESTED" if self.execution_origin == "OFFLINE_FIXTURE" else "NOT_RUN",
            "real_ibm_status": "REAL_IBM_EXECUTED" if self.real_hardware_executed else ("NOT_EXECUTED" if self.execution_origin == "OFFLINE_FIXTURE" else ("BLOCKED" if not self.access.get("runtime_available") or not self.access.get("credentials_detected") else "NOT_EXECUTED")),
            "overall_status": self.overall_status,
            "execution_status": self.execution_status,
            "execution_origin": self.execution_origin,
            "real_hardware_executed": self.real_hardware_executed,
            "backend": self.backend_name,
            "job_id": self.job_id,
            "shots": self.shots,
            "execution_timestamp": self.execution_timestamp,
            "execution_start_timestamp": self.execution_start_timestamp,
            "execution_end_timestamp": self.execution_end_timestamp,
            "ideal_distribution": _json_safe(self.ideal_distribution),
            "observed_counts": _json_safe(self.observed_counts),
            "observed_distribution": _json_safe(self.observed_distribution),
            "anomaly": _json_safe(self.anomaly),
            "diagnosis": _json_safe(self.diagnosis),
            "program_candidates": _json_safe(self.program_candidates),
            "hardware_candidates": _json_safe(self.hardware_candidates),
            "calibration_snapshot": _json_safe(self.calibration_snapshot),
            "calibration_snapshot_id": self.calibration_snapshot_id,
            "diagnosis_calibration_snapshot_id": self.calibration_snapshot_id if self.diagnosis is not None else None,
            "calibration_assessment": _json_safe(self.calibration_assessment),
            "transpilation_info": _json_safe(self.transpilation_info),
            "conversion_info": _json_safe(self.conversion_info),
            "access": _json_safe(self.access),
            "errors": list(self.errors),
            "metrics": {"accuracy": None, "precision": None, "recall": None, "f1": None, "reason": "No independent IBM execution ground truth was provided."},
        }


def _diagnosis_payload(result: DiagnosisResult) -> dict[str, Any]:
    return {
        "category": result.category.value,
        "anomaly_detected": result.anomaly_detected,
        "summary": result.summary,
        "evidence": list(result.evidence),
        "missing_evidence": list(result.missing_evidence),
    }


def _make_result(
    *, status: str, origin: str, backend_name: str | None, shots: int, access: IBMAccessStatus,
    ideal: dict[str, float] | None = None, counts: dict[str, int] | None = None,
    observed: dict[str, float] | None = None, diagnosis: DiagnosisResult | None = None,
    calibration: IBMCalibrationCapture | None = None, conversion: CircuitConversionResult | None = None,
    transpilation_info: dict[str, Any] | None = None, job_id: str | None = None,
    errors: list[str] | None = None, real_executed: bool = False,
    execution_start_timestamp: str | None = None, execution_end_timestamp: str | None = None,
    calibration_assessment: dict[str, Any] | None = None,
) -> IBMValidationResult:
    snapshot_id = _calibration_snapshot_id(calibration)
    assessment = dict(calibration_assessment or {})
    if calibration is not None:
        assessment.update({
            "backend_name": calibration.backend_name,
            "calibration_timestamp": calibration.timestamp,
            "calibration_retrieved_at": calibration.retrieved_at,
            "calibration_capture_source": calibration.capture_source,
            "calibration_snapshot_id": snapshot_id,
        })
    assessment.update({
        "execution_start_timestamp": execution_start_timestamp,
        "execution_end_timestamp": execution_end_timestamp,
    })
    program, hardware = _candidate_payload(diagnosis)
    stats = diagnosis.statistics_result if diagnosis else None
    anomaly = None if stats is None else {
        "detected": stats.anomaly_detected,
        "total_variation_distance": stats.total_variation_distance,
        "threshold": stats.threshold,
        "finite_shot_guard": stats.finite_shot_guard,
        "reason": stats.reason,
        "message": stats.message,
    }
    return IBMValidationResult(
        overall_status=status,
        execution_status=status,
        execution_origin=origin,
        backend_name=backend_name,
        job_id=job_id,
        shots=shots,
        execution_timestamp=utc_timestamp(),
        ideal_distribution=ideal,
        observed_counts=counts,
        observed_distribution=observed,
        anomaly=anomaly,
        diagnosis=_diagnosis_payload(diagnosis) if diagnosis else None,
        program_candidates=program,
        hardware_candidates=hardware,
        calibration_snapshot=calibration.to_dict() if calibration else None,
        transpilation_info=transpilation_info or {"status": "NOT_RUN"},
        conversion_info=({"status": conversion.status, **conversion.plan.to_dict(), "gate_mappings": conversion.gate_mappings, "error": conversion.error} if conversion else {"status": "NOT_RUN"}),
        access=access.to_dict(),
        errors=errors or [],
        real_hardware_executed=real_executed,
        execution_start_timestamp=execution_start_timestamp,
        execution_end_timestamp=execution_end_timestamp,
        calibration_snapshot_id=snapshot_id,
        calibration_assessment=assessment if assessment else None,
    )


def run_offline_validation(
    circuit: Sequence[Any] | None = None,
    *,
    shots: int = 256,
    seed: int = 1729,
    hardware_model: HardwareModel | None = None,
    observed_counts: Mapping[str, int] | None = None,
) -> IBMValidationResult:
    """Exercise planning/diagnosis deterministically without IBM or Qiskit access."""

    access = check_ibm_access_status()
    if not isinstance(shots, int) or shots <= 0:
        return _make_result(status="FAILED", origin="OFFLINE_FIXTURE", backend_name="offline-fixture", shots=0, access=access, errors=["shots must be a positive integer"])
    selected_circuit = list([("h", 0), ("measure", 0)] if circuit is None else circuit)
    plan = build_circuit_plan(selected_circuit)
    if not plan.supported:
        reason = "; ".join([*plan.errors, *(item["reason"] for item in plan.unsupported_operations)])
        conversion = CircuitConversionResult("UNSUPPORTED" if plan.unsupported_operations else "FAILED", None, plan, error=reason)
        return _make_result(status="FAILED", origin="OFFLINE_FIXTURE", backend_name="offline-fixture", shots=shots, access=access, conversion=conversion, errors=[reason])

    model = hardware_model or HardwareModel(plan.num_qubits, coupling_map=[])
    model_snapshot = model.to_calibration_snapshot()
    calibration = IBMCalibrationCapture(
        backend_name="offline-fixture",
        timestamp=utc_timestamp(),
        n_qubits=model.n_qubits,
        coupling_map=model.get_coupling_map(),
        backend_status={"operational": None, "status_msg": "offline deterministic fixture", "pending_jobs": None},
        qubits={
            qubit: {
                "t1": model.get_qubit_properties(qubit).t1,
                "t1_unit": "microseconds (model fixture)",
                "t2": model.get_qubit_properties(qubit).t2,
                "t2_unit": "microseconds (model fixture)",
                "readout_error": model.get_qubit_properties(qubit).readout_error,
            }
            for qubit in range(model.n_qubits)
        },
        gates=[{"gate": gate, **entry} for gate, entries in model_snapshot.gates.items() for entry in entries],
        unavailable_fields=[],
        hardware_model=model,
        retrieval_status="OFFLINE_FIXTURE",
        capture_source="OFFLINE_FIXTURE",
        retrieved_at=utc_timestamp(),
    )
    try:
        ideal = ideal_distribution(selected_circuit, plan.num_qubits, seed=seed)
    except Exception as exc:
        return _make_result(status="FAILED", origin="OFFLINE_FIXTURE", backend_name="offline-fixture", shots=shots, access=access, errors=[f"Ideal simulation failed: {type(exc).__name__}: {exc}"])

    fixture_counts = dict(observed_counts) if observed_counts is not None else _counts_from_probabilities(ideal, shots)
    try:
        observed = counts_to_probabilities(fixture_counts, shots=shots)
    except Exception as exc:
        return _make_result(status="FAILED", origin="OFFLINE_FIXTURE", backend_name="offline-fixture", shots=shots, ideal=ideal, counts=fixture_counts, access=access, errors=[f"Offline fixture counts invalid: {type(exc).__name__}: {exc}"])

    localization_circuit = _deterministic_localization_circuit(plan)
    try:
        diagnosis = diagnose_execution(
            expected_distribution=ideal,
            observed_distribution=observed,
            expected_circuit=localization_circuit,
            observed_circuit=localization_circuit,
            hardware_model=model,
            shots=shots,
        )
    except Exception as exc:
        return _make_result(status="FAILED", origin="OFFLINE_FIXTURE", backend_name="offline-fixture", shots=shots, ideal=ideal, counts=fixture_counts, observed=observed, access=access, calibration=calibration, errors=[f"Offline diagnosis failed: {type(exc).__name__}: {exc}"])
    conversion = CircuitConversionResult("OFFLINE_PLAN_ONLY", None, plan, error="Qiskit conversion/transpilation was not run in offline mode.")
    return _make_result(
        status="OFFLINE_TESTED",
        origin="OFFLINE_FIXTURE",
        backend_name="offline-fixture",
        shots=shots,
        access=access,
        ideal=ideal,
        counts=fixture_counts,
        observed=observed,
        diagnosis=diagnosis,
        calibration=calibration,
        conversion=conversion,
        transpilation_info={"status": "NOT_RUN_OFFLINE", "reason": "Offline fixtures are not IBM backend executions."},
        errors=[],
    )


def transpile_for_backend(circuit: Any, backend: Any, *, optimization_level: int = 1) -> dict[str, Any]:
    """Transpile independently; a failure here is not a diagnosis or hardware fault."""

    try:
        from qiskit import transpile
        transpiled = transpile(circuit, backend=backend, optimization_level=optimization_level)
        operation_counts = {str(name): int(count) for name, count in transpiled.count_ops().items()}
        target = getattr(backend, "target", None)
        backend_operations = getattr(target, "operation_names", None)
        if backend_operations is None:
            configuration = _value_or_call(backend, "configuration")
            backend_operations = getattr(configuration, "basis_gates", None)
        source_operations = {str(name): int(count) for name, count in circuit.count_ops().items()}
        layout_info = _extract_transpilation_layout(
            transpiled,
            logical_num_qubits=int(circuit.num_qubits),
            backend_num_qubits=int(getattr(backend, "num_qubits", transpiled.num_qubits)),
        )
        mapped_operations = []
        if backend_operations is not None:
            backend_operation_names = {str(name).lower() for name in backend_operations}
            for name in source_operations:
                if name.lower() not in backend_operation_names and name.lower() not in {"measure", "reset", "barrier"}:
                    mapped_operations.append({
                        "source": name,
                        "transpiled_operations": sorted(operation_counts),
                        "status": "TRANSPILER_MAPPED",
                    })
        return {
            "status": "TRANSPILED",
            "optimization_level": optimization_level,
            "original_operations": source_operations,
            "transpiled_operations": operation_counts,
            "backend_gate_mappings": mapped_operations,
            "depth": int(transpiled.depth()),
            "num_qubits": int(transpiled.num_qubits),
            **layout_info,
            "circuit_text": str(transpiled),
            "circuit": transpiled,
            "error": None,
        }
    except Exception as exc:
        unsupported = []
        try:
            source_names = circuit.count_ops()
            target = getattr(backend, "target", None)
            supported_names = getattr(target, "operation_names", None)
            if supported_names is not None:
                supported_names = {str(name).lower() for name in supported_names}
                unsupported = [str(name) for name in source_names if str(name).lower() not in supported_names and str(name).lower() not in {"measure", "reset", "barrier"}]
        except Exception:
            pass
        return {"status": "TRANSPILE_FAILED", "optimization_level": optimization_level, "unsupported_backend_gates": unsupported, "error": f"{type(exc).__name__}: {exc}", "circuit": None}


def _extract_transpilation_layout(
    transpiled: Any,
    *,
    logical_num_qubits: int,
    backend_num_qubits: int,
) -> dict[str, Any]:
    """Read Qiskit's final logical-to-physical mapping, rejecting ambiguous routes."""

    layout = getattr(transpiled, "layout", None)
    if layout is None:
        return {"layout_status": "MISSING_LAYOUT", "logical_to_physical": None, "layout_error": "Transpiler did not expose a layout."}
    # A routing permutation means logical states move during execution. A single
    # final map cannot safely attribute every original gate to one physical pair.
    final_layout = getattr(layout, "final_layout", None)
    if final_layout is not None:
        return {"layout_status": "ROUTING_LAYOUT_UNRESOLVED", "logical_to_physical": None, "layout_error": "Transpiler reports a final routing permutation; per-operation physical locations are unavailable."}
    final_index_layout = getattr(layout, "final_index_layout", None)
    if not callable(final_index_layout):
        return {"layout_status": "MISSING_LAYOUT", "logical_to_physical": None, "layout_error": "Layout has no final_index_layout mapping."}
    try:
        physical = final_index_layout(filter_ancillas=True)
    except TypeError:
        try:
            physical = final_index_layout()
        except Exception as exc:
            return {"layout_status": "INVALID_LAYOUT", "logical_to_physical": None, "layout_error": f"Could not read final layout: {type(exc).__name__}: {exc}"}
    except Exception as exc:
        return {"layout_status": "INVALID_LAYOUT", "logical_to_physical": None, "layout_error": f"Could not read final layout: {type(exc).__name__}: {exc}"}
    try:
        mapping = [int(index) for index in physical]
    except (TypeError, ValueError):
        mapping = []
    if len(mapping) != logical_num_qubits:
        return {"layout_status": "INVALID_LAYOUT", "logical_to_physical": None, "layout_error": f"Expected {logical_num_qubits} logical mappings, received {len(mapping)}."}
    if len(set(mapping)) != len(mapping) or any(index < 0 or index >= backend_num_qubits for index in mapping):
        return {"layout_status": "INVALID_LAYOUT", "logical_to_physical": None, "layout_error": "Layout contains duplicate or out-of-range physical qubit indices."}
    try:
        operation_counts = {str(name).lower(): int(count) for name, count in transpiled.count_ops().items()}
    except Exception:
        operation_counts = {}
    if operation_counts.get("swap", 0):
        return {"layout_status": "ROUTING_LAYOUT_UNRESOLVED", "logical_to_physical": None, "layout_error": "Transpiled circuit contains SWAP routing; per-operation physical locations are unavailable."}
    return {"layout_status": "MAPPED", "logical_to_physical": mapping, "layout_error": None}


def _map_plan_to_physical(plan: CircuitPlan, logical_to_physical: Sequence[int], backend_num_qubits: int) -> CircuitPlan:
    """Re-index every operation, including all operands of multi-qubit gates."""

    mapping = [int(index) for index in logical_to_physical]
    if len(mapping) != plan.num_qubits or len(set(mapping)) != len(mapping):
        raise ValueError("logical-to-physical mapping is missing or inconsistent")
    if any(index < 0 or index >= backend_num_qubits for index in mapping):
        raise ValueError("logical-to-physical mapping contains an out-of-range physical qubit")
    operations = [
        NormalizedOperation(
            operation.name,
            tuple(mapping[index] for index in operation.qubits),
            operation.parameters,
            operation.classical_bits,
        )
        for operation in plan.operations
    ]
    return CircuitPlan(backend_num_qubits, operations, list(plan.unsupported_operations), list(plan.errors))


def _job_identifier(job: Any) -> str | None:
    value = _value_or_call(job, "job_id")
    return str(value) if value is not None else None


def execute_on_ibm_backend(circuit: Sequence[Any], *, backend_name: str, token: str, instance: str | None = None, shots: int = 1024, channel: str = "ibm_quantum_platform", optimization_level: int = 1, max_calibration_age_seconds: float = DEFAULT_MAX_CALIBRATION_AGE_SECONDS) -> IBMValidationResult:
    """Transpile and submit a circuit only when runtime dependencies and credentials exist."""

    access = _explicit_credential_access_status(token)
    if isinstance(max_calibration_age_seconds, bool) or not isinstance(max_calibration_age_seconds, (int, float)) or not math.isfinite(float(max_calibration_age_seconds)) or max_calibration_age_seconds <= 0:
        raise ValueError("max_calibration_age_seconds must be a positive number")
    plan = build_circuit_plan(circuit)
    if not plan.supported:
        conversion = CircuitConversionResult(
            "UNSUPPORTED" if plan.unsupported_operations else "FAILED",
            None,
            plan,
            error="; ".join([*plan.errors, *(item["reason"] for item in plan.unsupported_operations)]),
        )
        return _make_result(status="FAILED", origin="REAL_IBM_REQUEST", backend_name=backend_name, shots=shots, access=access, conversion=conversion, errors=[conversion.error or "Circuit is unsupported"])
    if not isinstance(token, str) or not token.strip() or access.status != "ACCESS_CONFIGURED":
        return _make_result(status="BLOCKED", origin="REAL_IBM_REQUEST", backend_name=backend_name, shots=shots, access=access, errors=[access.reason or "IBM access is not configured"])
    if not isinstance(shots, int) or isinstance(shots, bool) or shots <= 0:
        return _make_result(status="FAILED", origin="REAL_IBM_REQUEST", backend_name=backend_name, shots=0, access=access, errors=["shots must be a positive integer"])

    try:
        ideal = ideal_distribution(circuit, plan.num_qubits)
    except Exception as exc:
        return _make_result(status="FAILED", origin="REAL_IBM_REQUEST", backend_name=backend_name, shots=shots, access=access, errors=[f"Ideal prediction failed before submission: {type(exc).__name__}: {exc}"])

    conversion = convert_to_qiskit_circuit(circuit)
    if conversion.status != "CONVERTED":
        status = "BLOCKED" if conversion.status == "BLOCKED" else "FAILED"
        return _make_result(status=status, origin="REAL_IBM_REQUEST", backend_name=backend_name, shots=shots, access=access, conversion=conversion, errors=[conversion.error or "Circuit conversion failed"])

    try:
        service = _runtime_service(token, instance, channel)
        backend = service.backend(backend_name)
    except Exception as exc:
        return _make_result(status="FAILED", origin="REAL_IBM_REQUEST", backend_name=backend_name, shots=shots, access=access, conversion=conversion, errors=[f"Backend selection failed: {type(exc).__name__}: {exc}"])

    capture = capture_backend_calibration(backend)
    calibration_assessment = calibration_freshness(capture, max_age_seconds=max_calibration_age_seconds)
    calibration_assessment.update({
        "completeness": "COMPLETE" if not capture.unavailable_fields and not capture.errors else "INCOMPLETE",
        "unavailable_fields": capture.unavailable_fields,
        "errors": capture.errors,
    })
    if capture.backend_name != backend_name:
        return _make_result(status="BLOCKED", origin="REAL_IBM_REQUEST", backend_name=backend_name, shots=shots, access=access, calibration=capture, conversion=conversion, errors=[f"Selected backend identity {capture.backend_name!r} does not match requested backend {backend_name!r}; no circuit was submitted."], calibration_assessment={**calibration_assessment, "backend_identity_status": "MISMATCH", "requested_backend": backend_name})
    calibration_assessment["backend_identity_status"] = "MATCH"
    if calibration_assessment["status"] != "FRESH":
        return _make_result(status="BLOCKED", origin="REAL_IBM_REQUEST", backend_name=capture.backend_name, shots=shots, access=access, calibration=capture, conversion=conversion, errors=[f"Calibration is {calibration_assessment['status'].lower()}; no circuit was submitted."], calibration_assessment=calibration_assessment)
    if not capture.backend_status or capture.backend_status.get("operational") is not True:
        backend_state = "unavailable" if capture.backend_status and capture.backend_status.get("operational") is False else "unknown"
        return _make_result(status="BLOCKED", origin="REAL_IBM_REQUEST", backend_name=capture.backend_name, shots=shots, access=access, calibration=capture, conversion=conversion, errors=[f"Backend operational status is {backend_state}; no circuit was submitted."], calibration_assessment={**calibration_assessment, "backend_operational": capture.backend_status.get("operational") if capture.backend_status else None})
    transpilation = transpile_for_backend(conversion.circuit, backend, optimization_level=optimization_level)
    if transpilation["status"] != "TRANSPILED":
        transpilation_info = {key: value for key, value in transpilation.items() if key != "circuit"}
        return _make_result(status="FAILED", origin="REAL_IBM_REQUEST", backend_name=capture.backend_name, shots=shots, access=access, calibration=capture, conversion=conversion, transpilation_info=transpilation_info, errors=[transpilation["error"] or "Transpilation failed"])

    transpilation_info = {key: value for key, value in transpilation.items() if key != "circuit"}
    if transpilation.get("layout_status") != "MAPPED":
        return _make_result(status="BLOCKED", origin="REAL_IBM_REQUEST", backend_name=capture.backend_name, shots=shots, access=access, calibration=capture, conversion=conversion, transpilation_info=transpilation_info, errors=[transpilation.get("layout_error") or "Physical qubit layout is unavailable; no circuit was submitted."], calibration_assessment={**calibration_assessment, "layout_status": transpilation.get("layout_status")})
    try:
        localization_plan = _map_plan_to_physical(plan, transpilation["logical_to_physical"], int(capture.n_qubits or 0))
    except (KeyError, TypeError, ValueError) as exc:
        return _make_result(status="BLOCKED", origin="REAL_IBM_REQUEST", backend_name=capture.backend_name, shots=shots, access=access, calibration=capture, conversion=conversion, transpilation_info=transpilation_info, errors=[f"Physical qubit layout is inconsistent; no circuit was submitted: {exc}"], calibration_assessment={**calibration_assessment, "layout_status": "INVALID_LAYOUT"})

    execution_start_timestamp = None
    execution_end_timestamp = None
    try:
        from qiskit_ibm_runtime.executor_sampler import Sampler

        sampler = Sampler(mode=backend)
        execution_start_timestamp = utc_timestamp()
        job = sampler.run([transpilation["circuit"]], shots=shots)
        job_id = _job_identifier(job)
        sampler_result = job.result()
        counts_raw = sampler_result[0].data.c.get_counts()
        observed = counts_to_probabilities(counts_raw, shots=shots)
        expected_widths = {len("".join(str(outcome).split())) for outcome in ideal}
        observed_widths = {len(outcome) for outcome in observed}
        if len(expected_widths) != 1 or observed_widths != expected_widths:
            raise ValueError("execution result measurement width is incompatible with the expected distribution")
        counts = {str(outcome): int(value) for outcome, value in counts_raw.items()}
        execution_end_timestamp = utc_timestamp()
        _validate_execution_interval(execution_start_timestamp, execution_end_timestamp)
    except Exception as exc:
        if execution_start_timestamp is not None:
            execution_end_timestamp = utc_timestamp()
        info = {key: value for key, value in transpilation.items() if key != "circuit"}
        return _make_result(status="FAILED", origin="REAL_IBM_REQUEST", backend_name=capture.backend_name, shots=shots, access=access, calibration=capture, conversion=conversion, transpilation_info=info, errors=[f"Hardware execution/result retrieval failed: {type(exc).__name__}: {exc}"], execution_start_timestamp=locals().get("execution_start_timestamp"), execution_end_timestamp=execution_end_timestamp, calibration_assessment={**calibration_assessment, "layout_status": "MAPPED"})

    model = capture.hardware_model
    if model is None:
        model = HardwareModel(max(capture.n_qubits or 1, conversion.plan.num_qubits))
    localization_circuit = _deterministic_localization_circuit(localization_plan)
    try:
        diagnosis = diagnose_execution(
            expected_distribution=ideal,
            observed_distribution=observed,
            expected_circuit=localization_circuit,
            observed_circuit=localization_circuit,
            hardware_model=model,
            shots=shots,
        )
    except Exception as exc:
        info = {key: value for key, value in transpilation.items() if key != "circuit"}
        return _make_result(status="FAILED", origin="REAL_IBM_REQUEST", backend_name=capture.backend_name, shots=shots, ideal=ideal, counts=counts, observed=observed, access=access, calibration=capture, conversion=conversion, transpilation_info=info, job_id=job_id, errors=[f"Diagnosis failed after hardware execution: {type(exc).__name__}: {exc}"], execution_start_timestamp=execution_start_timestamp, execution_end_timestamp=execution_end_timestamp, calibration_assessment={**calibration_assessment, "layout_status": "MAPPED"})
    if capture.unavailable_fields:
        calibration_gap = "Backend calibration snapshot is incomplete; unavailable fields: " + ", ".join(capture.unavailable_fields)
        if calibration_gap not in diagnosis.missing_evidence:
            diagnosis.missing_evidence.append(calibration_gap)
        if calibration_gap not in diagnosis.evidence:
            diagnosis.evidence.append(calibration_gap)
    transpilation_info = {key: value for key, value in transpilation.items() if key != "circuit"}
    return _make_result(status="REAL_IBM_EXECUTED", origin="REAL_IBM", backend_name=capture.backend_name, shots=shots, access=access, ideal=ideal, counts=counts, observed=observed, diagnosis=diagnosis, calibration=capture, conversion=conversion, transpilation_info=transpilation_info, job_id=job_id, real_executed=True, execution_start_timestamp=execution_start_timestamp, execution_end_timestamp=execution_end_timestamp, calibration_assessment={**calibration_assessment, "layout_status": "MAPPED"})


def _offline_report_result() -> IBMValidationResult:
    return run_offline_validation()


def build_ibm_validation_report(result: IBMValidationResult, discovery: Mapping[str, Any] | None = None) -> dict[str, Any]:
    report = result.to_dict()
    access = result.access
    report["backend_discovery"] = _json_safe(discovery or {"status": "NOT_RUN", "backends": []})
    report["calibration_retrieval_status"] = result.calibration_snapshot.get("retrieval_status") if result.calibration_snapshot else "NOT_RETRIEVED"
    report["phase_statuses"] = {
        "IMPLEMENTED": "IMPLEMENTED",
        "OFFLINE_TESTED": report["offline_status"],
        "REAL_IBM_EXECUTED": "REAL_IBM_EXECUTED" if result.real_hardware_executed else "NOT_EXECUTED",
        "BLOCKED": result.overall_status if result.overall_status == "BLOCKED" else ("BLOCKED" if not access.get("runtime_available") or not access.get("credentials_detected") else "NOT_BLOCKED"),
        "FAILED": "FAILED" if result.overall_status == "FAILED" else "NOT_FAILED",
    }
    return report


def write_ibm_validation_report(report: Mapping[str, Any], output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description="IBM Quantum validation adapter. Offline mode never contacts IBM.")
    parser.add_argument("--list-backends", action="store_true", help="Discover accessible IBM backends")
    parser.add_argument("--offline", action="store_true", help="Run deterministic local fixtures only")
    parser.add_argument("--backend", help="Backend name for a real IBM execution")
    parser.add_argument("--circuit", help="JSON file containing a universal circuit list")
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--channel", default="ibm_quantum_platform")
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent / "reports" / "ibm_validation_report.json"))
    args = parser.parse_args()

    if args.list_backends:
        # Preserve the legacy CLI behavior: --list-backends may use the configured
        # environment or saved Qiskit account. The public Python API never does.
        try:
            cli_service = _environment_runtime_service(args.channel)
            discovery = discover_ibm_backends(service=cli_service)
        except Exception as exc:
            access = check_ibm_access_status()
            discovery = {"status": "FAILED", "discovery_source": "IBM_RUNTIME", "backends": [], "access": access.to_dict(), "error": f"{type(exc).__name__}: {exc}"}
        result = run_offline_validation(shots=args.shots)
        report = build_ibm_validation_report(result, discovery)
    elif args.offline:
        result = run_offline_validation(shots=args.shots)
        report = build_ibm_validation_report(result)
    else:
        if not args.backend or not args.circuit:
            parser.error("real execution requires --backend and --circuit; use --offline for local validation")
        circuit_data = json.loads(Path(args.circuit).read_text(encoding="utf-8"))
        result = execute_on_ibm_backend(circuit_data, backend_name=args.backend, shots=args.shots, channel=args.channel)
        report = build_ibm_validation_report(result)

    write_ibm_validation_report(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


__all__ = [
    "IBMAccessStatus",
    "IBMCalibrationCapture",
    "IBMValidationResult",
    "CircuitPlan",
    "CircuitConversionResult",
    "NormalizedOperation",
    "build_circuit_plan",
    "build_ibm_validation_report",
    "calibration_freshness",
    "capture_backend_calibration",
    "check_ibm_access_status",
    "counts_to_probabilities",
    "convert_to_qiskit_circuit",
    "discover_ibm_backends",
    "execute_on_ibm_backend",
    "inspect_ibm_backend_calibration",
    "hardware_model_from_ibm_calibration",
    "ideal_distribution",
    "preflight_ibm_backend",
    "run_offline_validation",
    "transpile_for_backend",
    "write_ibm_validation_report",
]


if __name__ == "__main__":
    raise SystemExit(main())
