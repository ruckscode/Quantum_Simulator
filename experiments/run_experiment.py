"""Reproducible local execution and diagnosis for configured experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from diagnosis_engine import DiagnosisResult, diagnose_execution
from experiment_config import ExperimentConfig, load_experiment_config
from hardware_model import HardwareModel
from quantum_simulator import QuantumSimulator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_CONFIG_DIR = Path(__file__).resolve().parent / "configs"
DEFAULT_RESULT_DIR = PROJECT_ROOT / "reports" / "experiments"
DEFAULT_REPRODUCIBILITY_REPORT = PROJECT_ROOT / "reports" / "reproducibility_report.json"


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {item.name: _json_value(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _without_measurements(circuit: Sequence[Any]) -> list[Any]:
    result = []
    for operation in circuit:
        if isinstance(operation, Mapping):
            name = str(operation.get("name", operation.get("op", operation.get("gate", "")))).lower()
        elif isinstance(operation, (list, tuple)) and operation:
            name = str(operation[0]).lower()
        else:
            result.append(operation)
            continue
        if name != "measure":
            result.append(operation)
    return result


def _with_measurement(circuit: Sequence[Any]) -> list[Any]:
    operations = list(circuit)
    has_measurement = any(
        (str(operation.get("name", operation.get("op", operation.get("gate", "")))).lower() == "measure")
        if isinstance(operation, Mapping)
        else isinstance(operation, (tuple, list)) and bool(operation) and str(operation[0]).lower() == "measure"
        for operation in operations
    )
    return operations if has_measurement else [*operations, ("measure",)]


def _measurement_localization_view(circuit: Sequence[Any], num_qubits: int) -> list[Any]:
    """Replace stochastic measurement instructions by no-op barriers for localizers."""

    result: list[Any] = []
    for operation in circuit:
        if isinstance(operation, Mapping):
            name = str(operation.get("name", operation.get("op", operation.get("gate", "")))).lower()
            qubits = operation.get("qubits", [])
            if isinstance(qubits, int):
                qubits = [qubits]
            if name == "measure":
                result.append(("barrier", *(qubits or range(num_qubits))))
            else:
                result.append(operation)
        elif isinstance(operation, (tuple, list)) and operation and str(operation[0]).lower() == "measure":
            operands = list(operation[1:])
            if not operands:
                qubits = list(range(num_qubits))
            elif isinstance(operands[0], (list, tuple)):
                qubits = list(operands[0])
            else:
                qubits = [operands[0]]
            result.append(("barrier", *qubits))
        else:
            result.append(operation)
    return result


def _probabilities_from_counts(ideal_distribution: Mapping[str, float], counts: Mapping[str, int]) -> tuple[dict[str, float], dict[str, float]]:
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("observed counts must have a positive total")
    outcomes = set(ideal_distribution) | set(counts)
    widths = {len(outcome.replace(" ", "")) for outcome in outcomes}
    if len(widths) > 1:
        raise ValueError("ideal and observed outcomes have incompatible bit widths")
    expected = {outcome: float(ideal_distribution.get(outcome, 0.0)) for outcome in sorted(outcomes)}
    observed = {outcome: counts.get(outcome, 0) / total for outcome in sorted(outcomes)}
    return expected, observed


def _build_hardware_model(config: ExperimentConfig) -> HardwareModel | None:
    parameters = config.hardware_parameters
    if parameters is None:
        return None
    coupling_map = parameters.get("coupling_map", [])
    model = HardwareModel(
        config.num_qubits,
        coupling_map=coupling_map,
        backend_name=config.backend_name or "local-simulator",
        calibration_timestamp="synthetic-config",
    )
    qubits = parameters.get("qubits", {})
    if not isinstance(qubits, Mapping):
        raise ValueError("hardware_parameters.qubits must be an object")
    for qubit, properties in qubits.items():
        if not isinstance(properties, Mapping):
            raise ValueError(f"qubit properties for {qubit!r} must be an object")
        allowed = {key: properties[key] for key in ("t1", "t2", "readout_error", "status", "available") if key in properties}
        model.set_qubit_properties(int(qubit), **allowed)

    gates = parameters.get("gates", [])
    if isinstance(gates, Mapping):
        normalized_gates = []
        for name, entries in gates.items():
            if isinstance(entries, Mapping):
                entries = [entries]
            if not isinstance(entries, list):
                raise ValueError(f"gate entries for {name!r} must be an object or list")
            normalized_gates.extend({"gate": name, **entry} for entry in entries)
    elif isinstance(gates, list):
        normalized_gates = gates
    else:
        raise ValueError("hardware_parameters.gates must be a list or object")
    for entry in normalized_gates:
        if not isinstance(entry, Mapping) or "gate" not in entry:
            raise ValueError("each gate calibration must include a gate name")
        model.add_gate_calibration(
            str(entry["gate"]),
            qubits=entry.get("qubits", []),
            **{key: entry[key] for key in ("error", "duration", "supported") if key in entry},
        )

    edges = parameters.get("edges", [])
    if not isinstance(edges, list):
        raise ValueError("hardware_parameters.edges must be a list")
    for edge in edges:
        if not isinstance(edge, Mapping):
            raise ValueError("each edge calibration must be an object")
        pair = edge.get("qubits", edge.get("pair"))
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("each edge calibration must specify two qubits")
        model.add_edge(
            int(pair[0]),
            int(pair[1]),
            **{key: edge[key] for key in ("direction", "error", "duration", "supported") if key in edge},
        )
    return model


def _statevector_probabilities(circuit: Sequence[Any], num_qubits: int, seed: int) -> dict[str, float]:
    simulator = QuantumSimulator(num_qubits, seed=seed)
    simulator.run_circuit(_without_measurements(circuit))
    return {
        f"{index:0{num_qubits}b}": float(probability)
        for index, probability in enumerate(simulator.get_probabilities())
    }


def _measurement_targets(circuit: Sequence[Any], num_qubits: int) -> tuple[int, ...] | None:
    measurements: list[tuple[int, tuple[int, ...]]] = []
    for index, operation in enumerate(circuit):
        if isinstance(operation, Mapping):
            name = str(operation.get("name", operation.get("op", operation.get("gate", "")))).lower()
            raw_qubits = operation.get("qubits", [])
            if isinstance(raw_qubits, int):
                raw_qubits = [raw_qubits]
            qubits = tuple(int(value) for value in raw_qubits)
        elif isinstance(operation, (list, tuple)) and operation:
            name = str(operation[0]).lower()
            operands = list(operation[1:])
            if operands and isinstance(operands[0], (list, tuple)):
                qubits = tuple(int(value) for value in operands[0])
            elif operands and isinstance(operands[0], int):
                qubits = (int(operands[0]),)
            else:
                qubits = ()
        else:
            continue
        if name == "measure":
            measurements.append((index, qubits or tuple(range(num_qubits))))
    if not measurements:
        return tuple(range(num_qubits))
    if len(measurements) != 1:
        return None
    index, qubits = measurements[0]
    trailing = circuit[index + 1:]
    if any(
        not (isinstance(operation, (tuple, list)) and operation and str(operation[0]).lower() == "barrier")
        for operation in trailing
    ):
        return None
    if any(qubit < 0 or qubit >= num_qubits for qubit in qubits):
        raise ValueError("measurement references a qubit outside the configured circuit width")
    return qubits


def _marginal_ideal_distribution(
    statevector_distribution: Mapping[str, float],
    measured_qubits: tuple[int, ...] | None,
    fallback_counts: Mapping[str, int],
) -> dict[str, float]:
    if measured_qubits is None:
        total = sum(fallback_counts.values())
        if total <= 0:
            raise ValueError("seeded ideal simulator counts must have a positive total")
        return {key: value / total for key, value in sorted(fallback_counts.items())}

    width = len(measured_qubits)
    if width == 0:
        raise ValueError("measurement must target at least one qubit")
    result = {f"{index:0{width}b}": 0.0 for index in range(2**width)}
    for state, probability in statevector_distribution.items():
        output = "".join(state[qubit] for qubit in measured_qubits)
        result[output] += probability
    return result


@dataclass
class ExperimentResult:
    status: str
    experiment_metadata: dict[str, Any]
    configuration: dict[str, Any]
    circuit: dict[str, Any]
    ideal_distribution: dict[str, float] | None
    ideal_statevector_distribution: dict[str, float] | None
    simulator_observed_counts: dict[str, int] | None
    observed_counts: dict[str, int] | None
    observed_distribution: dict[str, float] | None
    anomaly: Any
    diagnosis: DiagnosisResult | None
    program_evidence: Any
    hardware_evidence: Any
    execution_metadata: dict[str, Any]
    reproducibility: dict[str, Any]
    errors: list[str]

    def _stable_payload(self) -> dict[str, Any]:
        payload = self.to_dict(include_fingerprint=False)
        payload["experiment_metadata"] = {
            key: value for key, value in payload["experiment_metadata"].items()
            if key != "created_at"
        }
        payload["execution_metadata"] = {
            key: value for key, value in payload["execution_metadata"].items()
            if key != "generated_at"
        }
        payload["reproducibility"] = {
            key: value for key, value in payload["reproducibility"].items()
            if key != "fingerprint"
        }
        return payload

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        data = {
            "status": self.status,
            "experiment_metadata": _json_value(self.experiment_metadata),
            "configuration": _json_value(self.configuration),
            "circuit": _json_value(self.circuit),
            "ideal_distribution": _json_value(self.ideal_distribution),
            "ideal_statevector_distribution": _json_value(self.ideal_statevector_distribution),
            "simulator_observed_counts": _json_value(self.simulator_observed_counts),
            "observed_counts": _json_value(self.observed_counts),
            "observed_distribution": _json_value(self.observed_distribution),
            "anomaly": _json_value(self.anomaly),
            "diagnosis": _json_value(self.diagnosis),
            "program_evidence": _json_value(self.program_evidence),
            "hardware_evidence": _json_value(self.hardware_evidence),
            "execution_metadata": _json_value(self.execution_metadata),
            "reproducibility": _json_value(self.reproducibility),
            "errors": list(self.errors),
        }
        if not include_fingerprint:
            data["reproducibility"].pop("fingerprint", None)
        return data

    def deterministic_fingerprint(self) -> str:
        stable = json.dumps(self._stable_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(stable.encode("utf-8")).hexdigest()


def _new_result(config: ExperimentConfig, **values: Any) -> ExperimentResult:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    errors = list(values.get("errors", []))
    status = values.get("status", "FAILED")
    result = ExperimentResult(
        status=status,
        experiment_metadata={
            "experiment_name": config.experiment_name,
            "validation_label": config.validation_label,
            "backend_name": config.backend_name,
            "created_at": timestamp,
        },
        configuration=config.to_dict(),
        circuit={
            "ideal": config.circuit,
            "observed": config.observed_circuit if config.observed_circuit is not None else config.circuit,
        },
        ideal_distribution=values.get("ideal_distribution"),
        ideal_statevector_distribution=values.get("ideal_statevector_distribution"),
        simulator_observed_counts=values.get("simulator_observed_counts"),
        observed_counts=values.get("observed_counts"),
        observed_distribution=values.get("observed_distribution"),
        anomaly=values.get("anomaly"),
        diagnosis=values.get("diagnosis"),
        program_evidence=values.get("program_evidence"),
        hardware_evidence=values.get("hardware_evidence"),
        execution_metadata={
            "execution_target": "LOCAL_SIMULATOR",
            "execution_kind": "SYNTHETIC_VALIDATION",
            "backend_api_used": False,
            "shots": config.shots,
            "random_seed": config.random_seed,
            "observation_source": values.get("observation_source", "NOT_EXECUTED"),
            "noise_parameters_applied": False,
            "generated_at": timestamp,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
        },
        reproducibility={
            "random_seed": config.random_seed,
            "simulator_version": "project-statevector-current",
            "deterministic_localizer_measurements": "measurement operations mapped to barriers for localization only",
            "fingerprint_excludes": ["execution_metadata.generated_at", "reproducibility.fingerprint"],
            "fingerprint": "",
        },
        errors=errors,
    )
    result.reproducibility["fingerprint"] = result.deterministic_fingerprint()
    return result


def run_experiment(
    config: ExperimentConfig | Mapping[str, Any] | str | Path,
    *,
    output_path: str | Path | None = None,
) -> ExperimentResult:
    """Run one local experiment using existing simulator and diagnosis APIs."""

    if isinstance(config, (str, Path)):
        resolved = load_experiment_config(config)
    elif isinstance(config, ExperimentConfig):
        resolved = config
    elif isinstance(config, Mapping):
        resolved = ExperimentConfig.from_mapping(config)
    else:
        raise TypeError("config must be an ExperimentConfig, mapping, or JSON path")

    observed_circuit = resolved.observed_circuit if resolved.observed_circuit is not None else resolved.circuit
    try:
        model = _build_hardware_model(resolved)
        ideal_statevector = _statevector_probabilities(resolved.circuit, resolved.num_qubits, resolved.random_seed)
        ideal_shots_circuit = _with_measurement(resolved.circuit)
        observed_shots_circuit = _with_measurement(observed_circuit)
        ideal_simulator = QuantumSimulator(resolved.num_qubits, seed=resolved.random_seed)
        observed_simulator = QuantumSimulator(resolved.num_qubits, seed=resolved.random_seed)
        ideal_counts = ideal_simulator.run_shots(ideal_shots_circuit, shots=resolved.shots, seed=resolved.random_seed)
        simulator_observed_counts = observed_simulator.run_shots(observed_shots_circuit, shots=resolved.shots, seed=resolved.random_seed)
        selected_observed_counts = dict(resolved.observed_counts) if resolved.observed_counts is not None else dict(simulator_observed_counts)
        ideal_distribution = _marginal_ideal_distribution(
            ideal_statevector,
            _measurement_targets(resolved.circuit, resolved.num_qubits),
            ideal_counts,
        )
        ideal_distribution, observed_distribution = _probabilities_from_counts(ideal_distribution, selected_observed_counts)

        localization_expected = _measurement_localization_view(resolved.circuit, resolved.num_qubits)
        localization_observed = _measurement_localization_view(observed_circuit, resolved.num_qubits)
        diagnosis = diagnose_execution(
            expected_distribution=ideal_distribution,
            observed_distribution=observed_distribution,
            expected_circuit=localization_expected,
            observed_circuit=localization_observed,
            hardware_model=model,
            threshold=resolved.anomaly_threshold,
            shots=resolved.shots,
        )
        result = _new_result(
            resolved,
            status="COMPLETED",
            ideal_distribution=ideal_distribution,
            ideal_statevector_distribution=ideal_statevector,
            simulator_observed_counts=simulator_observed_counts,
            observed_counts=selected_observed_counts,
            observed_distribution=observed_distribution,
            anomaly=diagnosis.statistics_result,
            diagnosis=diagnosis,
            program_evidence=diagnosis.program_result,
            hardware_evidence=diagnosis.hardware_result,
            observation_source="CONTROLLED_OBSERVATION_FIXTURE" if resolved.observed_counts is not None else "SEEDED_LOCAL_SIMULATOR_SHOTS",
        )
    except Exception as exc:
        result = _new_result(
            resolved,
            status="FAILED",
            errors=[f"{type(exc).__name__}: {exc}"],
            observation_source="FAILED",
        )

    if output_path is None:
        output_path = DEFAULT_RESULT_DIR / f"{_safe_name(resolved.experiment_name)}.json"
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _safe_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_")
    return normalized or "experiment"


def run_config_batch(
    config_paths: Sequence[str | Path],
    *,
    result_dir: str | Path = DEFAULT_RESULT_DIR,
    reproducibility_report_path: str | Path = DEFAULT_REPRODUCIBILITY_REPORT,
    test_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result_directory = Path(result_dir)
    entries = []
    for path in config_paths:
        config = load_experiment_config(path)
        first = run_experiment(config, output_path=result_directory / f"{_safe_name(config.experiment_name)}.json")
        repeated = run_experiment(config, output_path=result_directory / f"{_safe_name(config.experiment_name)}_repeat.json")
        entries.append({
            "configuration_file": str(Path(path)),
            "experiment_name": config.experiment_name,
            "seed": config.random_seed,
            "status": first.status,
            "result_file": str(result_directory / f"{_safe_name(config.experiment_name)}.json"),
            "execution_result": first.to_dict(),
            "repeated_run": {
                "status": repeated.status,
                "fingerprint": repeated.reproducibility["fingerprint"],
                "matched": first.reproducibility["fingerprint"] == repeated.reproducibility["fingerprint"],
            },
        })

    report = {
        "report_name": "Local Experiment Reproducibility Report",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "execution_target": "LOCAL_SIMULATOR",
        "validation_label": "SYNTHETIC_VALIDATION",
        "configurations_executed": [entry["experiment_name"] for entry in entries],
        "seeds": {entry["experiment_name"]: entry["seed"] for entry in entries},
        "results": entries,
        "repeated_runs_matched": all(entry["repeated_run"]["matched"] for entry in entries),
        "exact_test_summary": dict(test_summary or {}),
        "failures": [
            {"experiment_name": entry["experiment_name"], "errors": entry["execution_result"]["errors"]}
            for entry in entries
            if entry["status"] != "COMPLETED"
        ],
        "environment": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
        },
        "limitations": [
            "Experiments use the local simulator and controlled synthetic observation fixtures only.",
            "No Bugs4Q benchmark data or IBM hardware execution is used.",
            "Configured noise_parameters are recorded but not applied because this project has no noise model.",
        ],
    }
    destination = Path(reproducibility_report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a reproducible local quantum diagnosis experiment.")
    parser.add_argument("config", nargs="?", help="Path to one experiment JSON configuration")
    parser.add_argument("--all", action="store_true", help="Run all JSON configurations in experiments/configs")
    parser.add_argument("--output", help="Result JSON path for a single experiment")
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULT_DIR), help="Result directory for --all")
    parser.add_argument("--reproducibility-report", default=str(DEFAULT_REPRODUCIBILITY_REPORT))
    parser.add_argument("--tests-passed", type=int)
    parser.add_argument("--tests-failed", type=int, default=0)
    args = parser.parse_args()

    if args.all:
        paths = sorted(EXPERIMENT_CONFIG_DIR.glob("*.json"))
        if not paths:
            parser.error(f"no experiment JSON files found in {EXPERIMENT_CONFIG_DIR}")
        test_summary = None
        if args.tests_passed is not None:
            test_summary = {"passed": args.tests_passed, "failed": args.tests_failed}
        report = run_config_batch(
            paths,
            result_dir=args.result_dir,
            reproducibility_report_path=args.reproducibility_report,
            test_summary=test_summary,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if not report["failures"] and report["repeated_runs_matched"] else 1

    if not args.config:
        parser.error("provide a configuration path or use --all")
    result = run_experiment(args.config, output_path=args.output)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if result.status == "COMPLETED" else 1


__all__ = ["ExperimentResult", "run_config_batch", "run_experiment"]


if __name__ == "__main__":
    raise SystemExit(main())
