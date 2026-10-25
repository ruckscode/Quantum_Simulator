"""Generate the approved small, controlled experiment pilot (no ML training)."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from experiment_config import ExperimentConfig
from experiments.pilot import BASE_SEED, HARDWARE_READOUT_ERROR, PILOT_FAMILIES, REPETITIONS, SHOTS
from experiments.run_experiment import run_experiment


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "experiments" / "pilot"


def _profile(num_qubits: int, readout_error: float) -> dict[str, Any]:
    return {
        "coupling_map": [],
        "qubits": {
            str(qubit): {"t1": 1e6, "t2": 1e6, "readout_error": readout_error}
            for qubit in range(num_qubits)
        },
    }


def build_pilot_config(family_id: str, label: str, repetition: int) -> ExperimentConfig:
    family = PILOT_FAMILIES[family_id]
    reference = family["reference"]
    num_qubits = 2 if family_id in {"bell_pair", "controlled_pair"} else 1
    if label == "NORMAL":
        observed = reference
        hardware = None
        program_fault_id = None
        hardware_condition_id = None
        program_fault_description = None
        program_fault_operation_index = None
        program_fault_qubits = None
        hardware_affected_component = None
    elif label == "PROGRAM_FAULT":
        observed = family["mutated"]
        hardware = None
        program_fault_id = family["program_fault_id"]
        hardware_condition_id = None
        program_fault_description = family["program_fault_description"]
        program_fault_operation_index = family["program_fault_operation_index"]
        program_fault_qubits = family["program_fault_qubits"]
        hardware_affected_component = None
    elif label == "HARDWARE_ANOMALY":
        observed = reference
        readout_error = HARDWARE_READOUT_ERROR[family_id]
        hardware = _profile(num_qubits, readout_error)
        program_fault_id = None
        hardware_condition_id = f"readout_bit_flip_{int(readout_error * 100):02d}pct_v1"
        program_fault_description = None
        program_fault_operation_index = None
        program_fault_qubits = None
        hardware_affected_component = "qubit readout"
    else:
        raise ValueError(f"unsupported pilot label {label!r}")

    condition_index = {"NORMAL": 0, "PROGRAM_FAULT": 1, "HARDWARE_ANOMALY": 2}[label]
    family_index = list(PILOT_FAMILIES).index(family_id)
    seed = BASE_SEED + family_index * 100 + condition_index * 10 + repetition
    return ExperimentConfig(
        experiment_name=f"pilot_{family_id}_{label.lower()}_rep{repetition}",
        validation_label="SYNTHETIC_VALIDATION",
        num_qubits=num_qubits,
        shots=SHOTS,
        random_seed=seed,
        anomaly_threshold=0.05,
        circuit=reference,
        observed_circuit=observed,
        hardware_parameters=hardware,
        noise_parameters=None,
        observed_counts=None,
        ground_truth_label=label,
        circuit_family_id=family_id,
        program_fault_id=program_fault_id,
        hardware_condition_id=hardware_condition_id,
        program_fault_description=program_fault_description,
        program_fault_operation_index=program_fault_operation_index,
        program_fault_qubits=program_fault_qubits,
        hardware_affected_component=hardware_affected_component,
        repetition_index=repetition,
    )


def _distribution(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    return {key: value / total for key, value in counts.items()} if total else {}


def _tvd(left: dict[str, int], right: dict[str, int]) -> float:
    p, q = _distribution(left), _distribution(right)
    outcomes = set(p) | set(q)
    return 0.5 * sum(abs(p.get(outcome, 0.0) - q.get(outcome, 0.0)) for outcome in outcomes)


def _aggregate(reports: list[dict[str, Any]], family_id: str, label: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for report in reports:
        metadata = report["experiment_metadata"]
        if metadata["circuit_family_id"] == family_id and metadata["ground_truth_label"] == label:
            counts.update(report["observed_counts"] or {})
    return counts


def run_pilot(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_run_ids = {
        build_pilot_config(family_id, label, repetition).run_id
        for family_id in PILOT_FAMILIES
        for label in ("NORMAL", "PROGRAM_FAULT", "HARDWARE_ANOMALY")
        for repetition in range(REPETITIONS)
    }
    for old_report in output_dir.glob("run-*.json"):
        if old_report.stem not in expected_run_ids:
            old_report.unlink()
    reports: list[dict[str, Any]] = []
    for family_id in PILOT_FAMILIES:
        for label in ("NORMAL", "PROGRAM_FAULT", "HARDWARE_ANOMALY"):
            for repetition in range(REPETITIONS):
                config = build_pilot_config(family_id, label, repetition)
                result = run_experiment(config, output_path=output_dir / f"{config.run_id}.json")
                if result.status != "COMPLETED":
                    raise RuntimeError(f"pilot execution failed for {config.experiment_name}: {result.errors}")
                reports.append(result.to_dict())

    paired_effects = []
    for family_id in PILOT_FAMILIES:
        normal = _aggregate(reports, family_id, "NORMAL")
        program = _aggregate(reports, family_id, "PROGRAM_FAULT")
        hardware = _aggregate(reports, family_id, "HARDWARE_ANOMALY")
        paired_effects.append({
            "circuit_family_id": family_id,
            "program_fault_tvd_vs_normal": _tvd(normal, program),
            "hardware_tvd_vs_normal": _tvd(normal, hardware),
        })

    focus_family = "single_x"
    focus_normal = _aggregate(reports, focus_family, "NORMAL")
    focus_hardware = _aggregate(reports, focus_family, "HARDWARE_ANOMALY")
    focus_normal_reports = [r for r in reports if r["experiment_metadata"]["circuit_family_id"] == focus_family and r["experiment_metadata"]["ground_truth_label"] == "NORMAL"]
    focus_hardware_reports = [r for r in reports if r["experiment_metadata"]["circuit_family_id"] == focus_family and r["experiment_metadata"]["ground_truth_label"] == "HARDWARE_ANOMALY"]
    same_circuit_and_program = (
        {r["experiment_metadata"]["circuit_id"] for r in focus_normal_reports}
        == {r["experiment_metadata"]["circuit_id"] for r in focus_hardware_reports}
        and all(r["circuit"]["observed"] == focus_normal_reports[0]["circuit"]["observed"] for r in focus_hardware_reports)
    )
    focus_hardware_tvd = _tvd(focus_normal, focus_hardware)
    if not same_circuit_and_program or focus_hardware_tvd <= 0.10:
        raise RuntimeError("controlled single_x hardware-effect check did not confirm a hardware-driven distribution change")

    summary = {
        "pilot_name": "five_family_execution_derived_pilot",
        "records": len(reports),
        "circuit_families": len(PILOT_FAMILIES),
        "conditions": ["NORMAL", "PROGRAM_FAULT", "HARDWARE_ANOMALY"],
        "repetitions_per_group": REPETITIONS,
        "shots_per_run": SHOTS,
        "observed_counts_fixtures_used": False,
        "paired_observation_effects": paired_effects,
        "controlled_hardware_effect_check": {
            "circuit_family_id": focus_family,
            "same_reference_circuit_and_program": same_circuit_and_program,
            "normal_hardware_disabled": all(not r["execution_metadata"]["hardware_model_applied"] for r in focus_normal_reports),
            "hardware_profile_applied": all(r["execution_metadata"]["hardware_model_applied"] for r in focus_hardware_reports),
            "normal_aggregate_counts": dict(sorted(focus_normal.items())),
            "hardware_aggregate_counts": dict(sorted(focus_hardware.items())),
            "normal_vs_hardware_tvd": focus_hardware_tvd,
            "confirmed": True,
        },
        "hardware_execution_evidence": {
            "hardware_model_applied_in_reports": all(
                report["execution_metadata"]["hardware_model_applied"]
                for report in reports
                if report["experiment_metadata"]["ground_truth_label"] == "HARDWARE_ANOMALY"
            ),
            "normal_runs_hardware_disabled": all(
                not report["execution_metadata"]["hardware_model_applied"]
                for report in reports
                if report["experiment_metadata"]["ground_truth_label"] == "NORMAL"
            ),
        },
    }
    summary_path = output_dir / "pilot_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run_pilot(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
