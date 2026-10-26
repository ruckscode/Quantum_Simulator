"""Run a circuit through the simulator and write a diagnosis report.

Input is a JSON file describing the circuit, optional reference circuit, and
optional hardware calibration model. This is an offline simulator workflow; it
does not connect to IBM Quantum or claim physical-hardware measurements.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from quantum_simulator import diagnose_circuit
from quantum_simulator.hardware_model import HardwareModel
from quantum_simulator.hardware_data_provider import load_hardware_data


def json_value(value: Any) -> Any:
    """Convert diagnosis dataclasses and NumPy values to JSON-compatible data."""
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_hardware_model(config: dict[str, Any] | None) -> HardwareModel | None:
    if config is None:
        return None
    if not isinstance(config, dict):
        raise ValueError("'hardware_model' must be a JSON object")
    n_qubits = config.get("n_qubits")
    if isinstance(n_qubits, bool) or not isinstance(n_qubits, int) or n_qubits < 1:
        raise ValueError("hardware_model.n_qubits must be a positive integer")

    model = HardwareModel(
        n_qubits,
        coupling_map=None,
        backend_name=config.get("backend_name", "provided-calibration"),
        calibration_timestamp=config.get("calibration_timestamp"),
        calibration_timestamp_provenance=config.get(
            "calibration_timestamp_provenance",
            "reported" if config.get("calibration_timestamp") else "synthetic",
        ),
        default_gate_error=float(config.get("default_gate_error", 0.0)),
        default_gate_duration=float(config.get("default_gate_duration", 0.0)),
    )
    for item in config.get("qubits", []):
        props = dict(item)
        qubit = props.pop("qubit")
        model.set_qubit_properties(qubit, **props)
    for item in config.get("gates", []):
        entry = dict(item)
        gate = entry.pop("gate")
        qubits = entry.pop("qubits", [])
        error = entry.pop("error", 0.0)
        duration = entry.pop("duration", 0.0)
        supported = entry.pop("supported", True)
        model.add_gate_calibration(
            gate, qubits=qubits, error=error, duration=duration, supported=supported, **entry
        )
    for item in config.get("edges", []):
        if isinstance(item, dict):
            edge = dict(item)
            pair = edge.pop("qubits", edge.pop("pair", None))
            if not pair or len(pair) != 2:
                raise ValueError("Each hardware edge object needs 'qubits' or 'pair': [q1, q2]")
            model.add_edge(
                pair[0], pair[1],
                direction=edge.pop("direction", None),
                error=edge.pop("error", 0.0),
                duration=edge.pop("duration", 0.0),
                supported=edge.pop("supported", True),
                **edge,
            )
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            model.add_edge(item[0], item[1])
        else:
            raise ValueError("Each hardware edge must be a two-qubit pair or a calibration object")
    return model


def infer_circuit_width(*programs: list[Any] | None) -> int:
    """Infer the smallest logical register covering all referenced qubits."""
    highest = -1
    for program in programs:
        if not program:
            continue
        for instruction in program:
            if isinstance(instruction, dict):
                name = str(instruction.get("name") or instruction.get("op") or instruction.get("gate") or "").lower()
                qubits = instruction.get("qubits", [])
                if isinstance(qubits, int):
                    qubits = [qubits]
                values = qubits
            elif isinstance(instruction, (list, tuple)) and instruction:
                name = str(instruction[0]).lower()
                operands = list(instruction[1:])
                if name == "measure" and operands and isinstance(operands[0], (list, tuple)):
                    values = operands[0]
                elif name == "barrier":
                    values = operands
                else:
                    expected = 3 if name in {"ccx", "toffoli", "cswap", "fredkin"} else 2 if name in {"cx", "cnot", "cz", "swap", "ch", "cp", "crx", "cry", "crz", "rxx", "rzz"} else 1
                    values = operands[:expected]
            else:
                continue
            for value in values:
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    highest = max(highest, value)
    if highest < 0:
        raise ValueError("Could not infer circuit qubit count from the supplied instructions")
    return highest + 1


def build_report(config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError("Input JSON must be an object")
    if "circuit" not in config:
        raise ValueError("Input JSON requires a 'circuit' array")
    circuit = config["circuit"]
    reference = config.get("reference_circuit")
    if not isinstance(circuit, list) or (reference is not None and not isinstance(reference, list)):
        raise ValueError("'circuit' and 'reference_circuit' must be arrays of instructions")

    shots = config.get("shots", 1000)
    seed = config.get("seed", 42)
    threshold = config.get("threshold", 0.05)
    hardware_source_info = {
        "source": "user_supplied",
        "is_real_calibration": config.get("hardware_model", {}).get("calibration_timestamp_provenance") == "reported",
        "backend_name": config.get("hardware_model", {}).get("backend_name", "provided-calibration"),
        "calibration_timestamp": config.get("hardware_model", {}).get("calibration_timestamp"),
        "note": "Using calibration explicitly supplied in the input JSON.",
    }
    if config.get("hardware_model") is not None:
        hardware = load_hardware_model(config["hardware_model"])
    else:
        width = infer_circuit_width(circuit, reference)
        hardware, hardware_source_info = load_hardware_data(
            width,
            source=config.get("hardware_source", "auto"),
            backend_name=config.get("backend_name"),
            token=None,  # Never put API tokens in the input JSON or report.
            instance=config.get("ibm_instance"),
            seed=seed if isinstance(seed, int) and not isinstance(seed, bool) else 42,
        )
    result = diagnose_circuit(
        circuit,
        reference_circuit=reference,
        hardware_model=hardware,
        shots=shots,
        seed=seed,
        threshold=threshold,
        max_calibration_age_seconds=config.get("max_calibration_age_seconds"),
    )
    stats = result.statistics_result
    return {
        "report_name": "Quantum Circuit Diagnosis",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "circuit_name": config.get("name", "unnamed circuit"),
        "execution_mode": "local simulator",
        "physical_hardware_execution": False,
        "shots": shots,
        "seed": seed,
        "threshold": threshold,
        "circuit": json_value(circuit),
        "reference_circuit": json_value(reference),
        "hardware_model_supplied": hardware is not None,
        "hardware_source": json_value(hardware_source_info),
        "diagnosis": {
            "category": result.category.value,
            "anomaly_detected": result.anomaly_detected,
            "summary": result.summary,
            "total_variation_distance": stats.total_variation_distance if stats else None,
            "effective_threshold": max(stats.threshold, stats.finite_shot_guard) if stats else None,
            "program_candidates": json_value(result.program_evidence_candidates),
            "behavioral_program_candidates": json_value(result.program_behavioral_candidates),
            "trace_deviation_candidates": json_value(result.trace_deviation_candidates),
            "hardware_candidates": json_value(result.hardware_evidence_candidates),
            "evidence": result.evidence,
            "missing_evidence": result.missing_evidence,
            "calibration_freshness": json_value(result.calibration_freshness),
        },
        "limitations": [
            "Circuit execution is performed by the local simulator; calibration may be synthetic or fetched from IBM Quantum.",
            "Hardware attribution requires relevant, trustworthy calibration evidence.",
            "A source mismatch alone is not proof of a behavior-changing program fault.",
        ],
    }


def render_report(report: dict[str, Any]) -> str:
    diagnosis = report["diagnosis"]
    lines = [
        "=" * 64,
        "QUANTUM CIRCUIT DIAGNOSIS REPORT",
        "=" * 64,
        f"Circuit       : {report['circuit_name']}",
        f"Execution     : {report['execution_mode']}",
        f"Shots / seed  : {report['shots']} / {report['seed']}",
        f"Hardware source: {report.get('hardware_source', {}).get('source', 'not supplied')}",
        f"Backend        : {report.get('hardware_source', {}).get('backend_name', 'unknown')}",
        f"Calibration at : {report.get('hardware_source', {}).get('calibration_timestamp', 'not applicable')}",
        "",
        f"DIAGNOSIS: {diagnosis['category']}",
        f"Anomaly      : {diagnosis['anomaly_detected']}",
        f"Summary      : {diagnosis['summary']}",
        f"TVD          : {diagnosis['total_variation_distance']}",
        f"Threshold    : {diagnosis['effective_threshold']}",
        "",
        "Program fault candidates:",
    ]
    lines.extend(
        f"  step {item.get('index')}: {item.get('operation')} on {item.get('qubits')} — {item.get('reason')}"
        for item in diagnosis["program_candidates"]
    )
    if not diagnosis["program_candidates"]:
        lines.append("  none")
    lines.append("Hardware candidates:")
    lines.extend(
        f"  {item.get('component_type')} {item.get('component')} — {item.get('reason')}"
        for item in diagnosis["hardware_candidates"]
    )
    if not diagnosis["hardware_candidates"]:
        lines.append("  none supported by available calibration evidence")
    lines.append("Evidence:")
    lines.extend(f"  - {item}" for item in diagnosis["evidence"])
    lines.append("Missing evidence:")
    lines.extend(f"  - {item}" for item in diagnosis["missing_evidence"])
    if not diagnosis["missing_evidence"]:
        lines.append("  none reported")
    lines.extend(["", "Limitations:"])
    lines.extend(f"  - {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a quantum circuit and write a diagnosis report from the local simulator."
    )
    parser.add_argument("input", help="Path to circuit input JSON")
    parser.add_argument("--output", default="reports/diagnosis_report.json", help="JSON report destination")
    parser.add_argument("--text-output", default=None, help="Optional text report destination")
    args = parser.parse_args()

    try:
        config = json.loads(Path(args.input).read_text(encoding="utf-8"))
        report = build_report(config)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        text_report = render_report(report)
        text_output = Path(args.text_output) if args.text_output else output.with_suffix(".txt")
        text_output.parent.mkdir(parents=True, exist_ok=True)
        text_output.write_text(text_report, encoding="utf-8")
    except (OSError, json.JSONDecodeError, TypeError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    print(text_report)
    print(f"JSON report: {output}")
    print(f"Text report: {text_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
