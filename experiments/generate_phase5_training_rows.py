"""Generate only the frozen Phase 5 F1/F2 development rows and train its LR.

This workflow never loads dataset directories or reads F3/F4/F5 outputs.
All destinations are preflighted before the first simulator run and are
create-only; a partial run is preserved and requires manual review.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping, Sequence

from experiment_config import ExperimentConfig
from experiments.phase5_benchmark_infrastructure import (
    FAMILIES, PARTITIONS, SEEDS, SHOT_COUNTS, SUFFIXES,
    Case, build_ledger, observable_features, validate_calibration,
)
from experiments.phase5_model import train_phase5_artifact
from experiments.run_experiment import run_experiment

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "reports" / "experiments" / "phase5_training_f1_f2"
ROWS_PATH = ROOT / "data" / "phase5_training_f1_f2.jsonl"
ARTIFACT_PATH = ROOT / "src" / "quantum_simulator" / "models" / "observable_logistic_regression.json"
ALLOWED_FAMILIES = PARTITIONS["development"]
ALLOWED_SUFFIXES = tuple(suffix for suffix in SUFFIXES if suffix != "C15")
EXPECTED_LABEL = {"N0": "NORMAL", "P1": "PROGRAM_FAULT", "P2": "PROGRAM_FAULT"}
G_PROBABILITY = {"G02": .02, "G06": .06, "G15": .15, "G30": .30}
R_PROBABILITY = {"R02": .02, "R06": .06, "R15": .15, "R30": .30}
R_TARGET = {FAMILIES[0]: 2, FAMILIES[1]: 2}
G_TARGET = {FAMILIES[0]: ("cx", (2, 3)), FAMILIES[1]: ("ccx", (0, 1, 2))}


def _expected_config_keys() -> set[tuple[str, str, int, int]]:
    return {
        (family, suffix, seed, shots)
        for family in sorted(ALLOWED_FAMILIES)
        for suffix in ALLOWED_SUFFIXES
        for seed in SEEDS
        for shots in SHOT_COUNTS
    }


def validate_training_record(record: Mapping) -> None:
    """Validate audit identity and single-class truth before feature projection."""
    required = {"family_id", "suffix", "case_id", "seed", "shots", "label"}
    if not isinstance(record, Mapping) or set(record) != required:
        raise ValueError("training audit records must contain exactly family/case/seed/shot/label fields")
    family, suffix = record["family_id"], record["suffix"]
    if family not in ALLOWED_FAMILIES:
        raise ValueError("training data may contain only F1/F2 development families")
    if suffix not in ALLOWED_SUFFIXES or suffix == "C15":
        raise ValueError("training data contains an excluded or unsupported case suffix")
    if record["case_id"] != f"{family}-{suffix}":
        raise ValueError("case_id does not match its validated family and suffix")
    label = EXPECTED_LABEL.get(suffix, "HARDWARE_ANOMALY" if suffix.startswith(("R", "G")) else None)
    if label is None or record["label"] != label or isinstance(record["label"], (tuple, frozenset, set, list)):
        raise ValueError("training label is invalid or not a supported single-class truth")
    key = (family, suffix, record["seed"], record["shots"])
    if key not in _expected_config_keys():
        raise ValueError("unexpected family/case/seed/shot training configuration")


def validate_training_records(records: Sequence[Mapping]) -> None:
    """Fail closed on wrong family/case/label or incomplete/duplicate grid."""
    expected = _expected_config_keys()
    found: list[tuple[str, str, int, int]] = []
    for record in records:
        validate_training_record(record)
        family, suffix = record["family_id"], record["suffix"]
        key = (family, suffix, record["seed"], record["shots"])
        if key not in expected:
            raise ValueError("unexpected family/case/seed/shot training configuration")
        found.append(key)
    if len(found) != len(set(found)):
        raise ValueError("duplicate training configurations")
    if set(found) != expected or len(found) != len(expected):
        raise ValueError("training configuration grid is missing expected configurations")


def _profile(case: Case) -> dict:
    readout_p = R_PROBABILITY.get(case.suffix, 0.0)
    readout_target = R_TARGET[case.family_id]
    qubits = {
        q: {
            "readout_error": readout_p if case.suffix.startswith("R") and q == readout_target else 0.0,
            "t1": 1e6, "t2": 1e6, "status": "available", "available": True,
        }
        for q in range(4)
    }
    gate_target, gate_qubits = G_TARGET[case.family_id]
    gate_error = G_PROBABILITY.get(case.suffix, 0.0)
    keys = set()
    for circuit in (case.expected_circuit, case.observed_circuit):
        for name, *operands in circuit:
            keys.add((name, tuple(operands)))
    gates = [
        {
            "gate": name,
            "qubits": list(qubits_tuple),
            "error": gate_error if case.suffix.startswith("G") and (name, qubits_tuple) == (gate_target, gate_qubits) else 0.0,
            "duration": 0.0,
            "supported": True,
        }
        for name, qubits_tuple in sorted(keys)
    ]
    return {"qubits": qubits, "gates": gates, "coupling_map": [], "edges": []}


def _config(case: Case, seed: int, shots: int) -> ExperimentConfig:
    label = EXPECTED_LABEL.get(case.suffix, "HARDWARE_ANOMALY")
    program_fault = label == "PROGRAM_FAULT"
    hardware_fault = label == "HARDWARE_ANOMALY"
    name = f"phase5train_{case.case_id}_seed{seed}_shots{shots}"
    return ExperimentConfig(
        experiment_name=name,
        num_qubits=4,
        shots=shots,
        circuit=[list(op) for op in case.expected_circuit],
        observed_circuit=[list(op) for op in case.observed_circuit],
        random_seed=seed,
        anomaly_threshold=.05,
        hardware_parameters=_profile(case),
        ground_truth_label=label,
        circuit_family_id=case.family_id,
        program_fault_id=case.suffix if program_fault else None,
        program_fault_description=case.suffix if program_fault else None,
        hardware_condition_id=case.suffix if hardware_fault else None,
        hardware_affected_component=("READOUT_QUBIT" if case.suffix.startswith("R") else "GATE_TUPLE") if hardware_fault else None,
    )


def _write_rows_create_only(rows: Sequence[dict]) -> None:
    temporary = ROWS_PATH.with_name(ROWS_PATH.name + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    try:
        os.link(temporary, ROWS_PATH)
    finally:
        temporary.unlink(missing_ok=True)


def _preflight_destinations() -> None:
    if RESULT_DIR.exists():
        raise FileExistsError(f"refusing to use existing Phase 5 result directory: {RESULT_DIR}")
    for path in (ROWS_PATH, ROWS_PATH.with_name(ROWS_PATH.name + ".tmp"),
                 ARTIFACT_PATH, ARTIFACT_PATH.with_name(ARTIFACT_PATH.name + ".tmp")):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing Phase 5 output: {path}")


def generate_and_train() -> dict:
    """Run the exact 330-row F1/F2 grid, persist results/rows, then train LR."""
    _preflight_destinations()
    cases = [case for case in build_ledger()
             if case.family_id in ALLOWED_FAMILIES and case.suffix in ALLOWED_SUFFIXES]
    if len(cases) != 22 or any(case.suffix == "C15" for case in cases):
        raise ValueError("eligible Phase 5 development template set is not exactly the frozen 22 cases")
    for case in cases:
        validate_calibration(_profile(case), case.expected_circuit, case.observed_circuit, case=case)
    planned = [
        {"family_id": case.family_id, "suffix": case.suffix, "case_id": case.case_id,
         "seed": seed, "shots": shots,
         "label": EXPECTED_LABEL.get(case.suffix, "HARDWARE_ANOMALY")}
        for case in cases for seed in SEEDS for shots in SHOT_COUNTS
    ]
    validate_training_records(planned)

    RESULT_DIR.mkdir(parents=True, exist_ok=False)
    audit_records, training_rows = [], []
    for case in cases:
        for seed in SEEDS:
            for shots in SHOT_COUNTS:
                config = _config(case, seed, shots)
                result_path = RESULT_DIR / f"{case.case_id}__seed{seed}__shots{shots}.json"
                if result_path.exists():
                    raise FileExistsError(f"refusing to overwrite result file: {result_path}")
                result = run_experiment(config, output_path=result_path)
                if result.status != "COMPLETED" or result.errors:
                    raise RuntimeError(f"simulator configuration failed: {case.case_id}, seed={seed}, shots={shots}: {result.errors}")
                label = EXPECTED_LABEL.get(case.suffix, "HARDWARE_ANOMALY")
                if label not in ("NORMAL", "PROGRAM_FAULT", "HARDWARE_ANOMALY"):
                    raise ValueError(f"unsupported single-class training truth for {case.case_id}")
                audit_records.append({
                    "family_id": case.family_id, "suffix": case.suffix,
                    "case_id": case.case_id, "seed": seed, "shots": shots, "label": label,
                })
                validate_training_record(audit_records[-1])
                features = observable_features(result.ideal_distribution, result.observed_distribution)
                training_rows.append({"family_id": case.family_id, "features": features, "label": label})

    validate_training_records(audit_records)
    if len(training_rows) != 330:
        raise ValueError(f"expected exactly 330 training rows, found {len(training_rows)}")
    _write_rows_create_only(training_rows)
    artifact = train_phase5_artifact(training_rows, ARTIFACT_PATH)
    return {"configuration_count": len(audit_records), "row_count": len(training_rows),
            "raw_result_count": len(list(RESULT_DIR.glob("*.json"))),
            "shot_circuit_executions": 1_182_720,
            "artifact_path": str(ARTIFACT_PATH),
            "training_families": artifact["metadata"]["training_families"]}


if __name__ == "__main__":
    print(json.dumps(generate_and_train(), indent=2, sort_keys=True))
