"""Build and execute a fixed, diverse 30-family controlled experiment set."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np

from experiment_config import ExperimentConfig
from experiments.run_experiment import _source_fingerprints, run_experiment


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "experiments" / "full_dataset"
FAMILY_COUNT = 30
REPETITIONS = 3
SHOTS = 1024
EXECUTION_SEED_BASE = 930_000
DESIGN_SEED_BASE = 6_104_271
LABELS = ("NORMAL", "PROGRAM_FAULT", "HARDWARE_ANOMALY")
SINGLE_GATES = ("h", "x", "y", "z", "s", "t", "rx", "ry", "rz")
ROTATIONS = {"rx", "ry", "rz"}
TWO_QUBIT_GATES = ("cx", "cz", "swap", "rxx")
ANGLES = (0.29, 0.61, 0.93, 1.27, 1.83, 2.41)


def _family_name(index: int) -> str:
    return f"circuit_family_{index + 1:02d}"


def _operation_qubits(operation: list[Any]) -> list[int]:
    if not operation:
        return []
    name, operands = str(operation[0]).lower(), operation[1:]
    if name in TWO_QUBIT_GATES or name in {"cp", "crx", "cry", "crz"}:
        return [int(operands[0]), int(operands[1])]
    if name in {"ccx", "toffoli", "cswap", "fredkin"}:
        return [int(value) for value in operands[:3]]
    if name == "measure":
        if not operands:
            return []
        return [int(value) for value in operands[0]] if isinstance(operands[0], (list, tuple)) else [int(operands[0])]
    return [int(operands[0])] if operands and isinstance(operands[0], int) else []


def _reference_circuit(index: int) -> tuple[int, list[list[Any]], list[int]]:
    """Construct a deterministic, structurally varied reference circuit."""
    design_seed = DESIGN_SEED_BASE + index
    num_qubits = 1 + ((index * 7 + index // 3) % 4)
    depth = 2 + ((index * 5 + 1) % 6)
    operations: list[list[Any]] = []
    for layer in range(depth):
        for offset in range(num_qubits):
            qubit = (offset + layer + index) % num_qubits
            gate = SINGLE_GATES[(index * 3 + layer * 2 + qubit) % len(SINGLE_GATES)]
            if gate in ROTATIONS:
                angle = ANGLES[(index + 2 * layer + qubit) % len(ANGLES)]
                operations.append([gate, qubit, angle])
            else:
                operations.append([gate, qubit])

    # A further parameterized operation gives every family a reproducible
    # parameter-change candidate, with its qubit and angle varied by family.
    rotation = ("rx", "ry", "rz")[(index + 1) % 3]
    rotation_qubit = (index * 3 + 1) % num_qubits
    # A non-repeating family-specific parameter prevents structurally similar
    # templates from collapsing to the same circuit identity.
    rotation_angle = 0.15 + ((index * 0.61803398875) % 2.5)
    operations.append([rotation, rotation_qubit, rotation_angle])

    edges: list[tuple[int, int]] = []
    if num_qubits > 1:
        topology = index % 4
        if topology == 0:  # linear chain
            edges = [(q, q + 1) for q in range(num_qubits - 1)]
        elif topology == 1:  # star
            edges = [(0, q) for q in range(1, num_qubits)]
        elif topology == 2:  # ring, with no duplicate edge for two qubits
            edges = [(q, q + 1) for q in range(num_qubits - 1)]
            if num_qubits > 2:
                edges.append((num_qubits - 1, 0))
        else:  # disjoint nearest-neighbor pairs
            edges = [(q, q + 1) for q in range(0, num_qubits - 1, 2)]
        edge_gate = TWO_QUBIT_GATES[(index // 2) % len(TWO_QUBIT_GATES)]
        for edge_index, (left, right) in enumerate(edges):
            if edge_gate == "rxx":
                operations.append([edge_gate, left, right, ANGLES[(index + edge_index) % len(ANGLES)]])
            else:
                operations.append([edge_gate, left, right])

    # Use full-register, one-qubit-subset, and two-qubit-subset measurements.
    if index % 3 == 0:
        measured = list(range(num_qubits))
    elif index % 3 == 1:
        measured = [0]
    else:
        measured = sorted({0, num_qubits - 1})
    operations.append(["measure", measured])
    return num_qubits, operations, measured


def _mutate_program(index: int, circuit: list[list[Any]], measured: list[int]) -> tuple[list[list[Any]], str, int, list[int], str]:
    mutated = copy.deepcopy(circuit)
    body = mutated[:-1]
    mode = index % 5

    if mode == 0:  # gate replacement
        candidates = [position for position, op in enumerate(body) if op[0] in SINGLE_GATES]
        position = candidates[-1]
        old = body[position]
        replacement = "x" if old[0] != "x" else "h"
        new_op = [replacement, old[1]]
        body[position] = new_op
        qubits = _operation_qubits(old)
        description = f"Replace operation {position}: {old} with {new_op}."
        kind = "gate_replacement"
    elif mode == 1:  # deletion of an explicitly identified single-qubit gate
        candidates = [position for position, op in enumerate(body) if op[0] in SINGLE_GATES]
        position = candidates[-1]
        old = body.pop(position)
        qubits = _operation_qubits(old)
        description = f"Delete operation {position}: {old}."
        kind = "gate_deletion"
    elif mode == 2:  # parameter modification
        candidates = [position for position, op in enumerate(body) if op[0] in ROTATIONS]
        position = candidates[-1]
        old = list(body[position])
        body[position][2] = float(body[position][2] + 0.47)
        qubits = _operation_qubits(old)
        description = f"Modify operation {position} parameter from {old[2]:.8g} to {body[position][2]:.8g}."
        kind = "parameter_modification"
    elif mode == 3 and len(measured) > 1:  # controlled two-qubit gate replacement
        candidates = [position for position, op in enumerate(body) if op[0] in TWO_QUBIT_GATES]
        if candidates:
            position = candidates[-1]
            old = list(body[position])
            replacement = "cx" if old[0] != "cx" else "cz"
            new_op = [replacement, *old[1:3]]
            body[position] = new_op
            qubits = _operation_qubits(old)
            description = f"Replace operation {position}: {old} with {new_op}."
            kind = "two_qubit_gate_replacement"
        else:
            position = len(body)
            qubits = [measured[0]]
            body.append(["x", measured[0]])
            description = f"Insert X on measured qubit {measured[0]} before measurement."
            kind = "gate_insertion"
    elif mode == 3:  # one-qubit families still receive an explicit program mutation
        position = len(body)
        qubits = [measured[0]]
        body.append(["x", measured[0]])
        description = f"Insert X on measured qubit {measured[0]} before measurement."
        kind = "gate_insertion"
    else:  # gate insertion
        position = len(body)
        qubits = [measured[0]]
        body.append(["x", measured[0]])
        description = f"Insert X on measured qubit {measured[0]} before measurement."
        kind = "gate_insertion"

    mutated = [*body, mutated[-1]]
    return mutated, kind, position, qubits, description


def _hardware_profile(index: int, num_qubits: int, circuit: list[list[Any]]) -> tuple[dict[str, Any], str, str]:
    qubits = {
        str(q): {"t1": 1e6, "t2": 1e6, "readout_error": 0.0}
        for q in range(num_qubits)
    }
    profile: dict[str, Any] = {"coupling_map": [], "qubits": qubits, "gates": []}
    affected_q = index % num_qubits
    family_band = (index // 4) % 6
    mode = index % 4
    if mode == 0:
        error = (0.12, 0.18, 0.24, 0.30, 0.36, 0.42)[family_band]
        qubits[str(affected_q)]["readout_error"] = error
        component = f"readout on qubit {affected_q}"
        parameters = f"readout_error={error:.2f}"
    elif mode == 1:
        operation = next(op for op in circuit[:-1] if op[0] in SINGLE_GATES)
        gate = operation[0]
        gate_qubits = _operation_qubits(operation)
        error = (0.08, 0.12, 0.16, 0.20, 0.24, 0.28)[family_band]
        profile["gates"].append({"gate": gate, "qubits": gate_qubits, "error": error, "duration": 0.4})
        component = f"{gate} gate on qubits {gate_qubits}"
        parameters = f"gate_error={error:.2f}, duration=0.4"
    elif mode == 2:
        t1 = (80.0, 64.0, 52.0, 44.0, 36.0, 30.0)[family_band]
        t2 = min(t1 * 0.75, (60.0, 48.0, 39.0, 33.0, 27.0, 22.0)[family_band])
        qubits[str(affected_q)]["t1"] = t1
        qubits[str(affected_q)]["t2"] = t2
        operation = next((op for op in circuit[:-1] if affected_q in _operation_qubits(op)), circuit[0])
        profile["gates"].append({"gate": operation[0], "qubits": _operation_qubits(operation), "error": 0.0, "duration": 8.0})
        component = f"relaxation on qubit {affected_q}"
        parameters = f"t1={t1:.1f}, t2={t2:.1f}, gate_duration=8.0"
    else:
        readout_error = (0.12, 0.16, 0.20, 0.24, 0.28, 0.32)[family_band]
        operation = next(op for op in circuit[:-1] if op[0] in SINGLE_GATES)
        gate_error = (0.06, 0.10, 0.14, 0.18, 0.22, 0.26)[family_band]
        qubits[str(affected_q)]["readout_error"] = readout_error
        gate_qubits = _operation_qubits(operation)
        profile["gates"].append({"gate": operation[0], "qubits": gate_qubits, "error": gate_error, "duration": 0.4})
        component = f"readout qubit {affected_q} and {operation[0]} gate on qubits {gate_qubits}"
        parameters = f"readout_error={readout_error:.2f}, gate_error={gate_error:.2f}, duration=0.4"

    canonical = json.dumps({"profile": profile, "component": component}, sort_keys=True, separators=(",", ":"))
    condition_id = "hardware-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return profile, component, condition_id


def build_family_configs(index: int) -> list[ExperimentConfig]:
    family_id = _family_name(index)
    num_qubits, reference, measured = _reference_circuit(index)
    mutated, mutation_kind, op_index, affected_qubits, mutation_description = _mutate_program(index, reference, measured)
    healthy_profile = {
        "coupling_map": [],
        "qubits": {str(q): {"t1": 1e6, "t2": 1e6, "readout_error": 0.0} for q in range(num_qubits)},
        "gates": [],
    }
    hardware_profile, hardware_component, hardware_id = _hardware_profile(index, num_qubits, reference)
    profile_by_label = {
        "NORMAL": (reference, healthy_profile, None, None, None, None, None, None),
        "PROGRAM_FAULT": (mutated, healthy_profile, f"fault-{index + 1:02d}-{mutation_kind}", None, mutation_description, op_index, affected_qubits, None),
        "HARDWARE_ANOMALY": (reference, hardware_profile, None, hardware_id, None, None, None, hardware_component),
    }
    family_configs = []
    for condition_index, label in enumerate(LABELS):
        observed, profile, fault_id, hw_id, fault_description, affected_op, fault_qubits, component = profile_by_label[label]
        for repetition in range(REPETITIONS):
            seed = EXECUTION_SEED_BASE + index * 1_000 + condition_index * 100 + repetition
            config = ExperimentConfig(
                experiment_name=f"full_{family_id}_{label.lower()}_rep{repetition}",
                validation_label="SYNTHETIC_VALIDATION",
                num_qubits=num_qubits,
                shots=SHOTS,
                random_seed=seed,
                anomaly_threshold=0.05,
                circuit=reference,
                observed_circuit=observed,
                hardware_parameters=profile,
                noise_parameters=None,
                observed_counts=None,
                description=f"Fixed design seed {DESIGN_SEED_BASE + index}; {label} controlled condition.",
                ground_truth_label=label,
                circuit_family_id=family_id,
                # Keep parameterized siblings from the same deterministic
                # design recipe together during held-out evaluation.
                circuit_lineage_id=f"design_recipe_{index % 6 + 1:02d}",
                program_fault_id=fault_id,
                hardware_condition_id=hw_id,
                repetition_index=repetition,
                program_fault_description=fault_description,
                program_fault_operation_index=affected_op,
                program_fault_qubits=fault_qubits,
                hardware_affected_component=component,
            )
            family_configs.append(config)
    return family_configs


def generate_dataset(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = [config for family_index in range(FAMILY_COUNT) for config in build_family_configs(family_index)]
    expected_run_ids = {config.run_id for config in configs}
    for old_report in output_dir.glob("run-*.json"):
        if old_report.stem not in expected_run_ids:
            old_report.unlink()

    succeeded = 0
    for config in configs:
        result = run_experiment(config, output_path=output_dir / f"{config.run_id}.json")
        if result.status != "COMPLETED":
            raise RuntimeError(f"execution failed for {config.experiment_name}: {result.errors}")
        succeeded += 1

    manifest = {
        "dataset_name": "controlled_execution_derived_30_family_dataset",
        "design_seed_base": DESIGN_SEED_BASE,
        "execution_seed_base": EXECUTION_SEED_BASE,
        "families": [
            {
                "circuit_family_id": _family_name(index),
                "circuit_lineage_id": f"design_recipe_{index % 6 + 1:02d}",
                "circuit_id": build_family_configs(index)[0].circuit_id,
                "num_qubits": build_family_configs(index)[0].num_qubits,
                "circuit_depth_operations": len(build_family_configs(index)[0].circuit) - 1,
                "program_fault_id": build_family_configs(index)[3].program_fault_id,
                "program_fault_description": build_family_configs(index)[3].program_fault_description,
                "hardware_condition_id": build_family_configs(index)[6].hardware_condition_id,
                "hardware_affected_component": build_family_configs(index)[6].hardware_affected_component,
                "hardware_parameters": build_family_configs(index)[6].hardware_parameters,
            }
            for index in range(FAMILY_COUNT)
        ],
        "records_written": succeeded,
        "repetitions_per_group": REPETITIONS,
        "shots_per_run": SHOTS,
        "fixed_observed_counts_used": False,
        "generation_provenance": generation_provenance(),
    }
    destination = output_dir / "dataset_manifest.json"
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "families"}, indent=2, sort_keys=True))
    print(f"circuit families: {FAMILY_COUNT}; unique circuit designs: {len({row['circuit_id'] for row in manifest['families']})}")
    print(f"qubit-count range: {min(row['num_qubits'] for row in manifest['families'])}-{max(row['num_qubits'] for row in manifest['families'])}")
    print(f"operation-depth range: {min(row['circuit_depth_operations'] for row in manifest['families'])}-{max(row['circuit_depth_operations'] for row in manifest['families'])}")
    print(f"hardware profiles: {len({row['hardware_condition_id'] for row in manifest['families']})}")
    return manifest


def generation_provenance() -> dict[str, Any]:
    """Identify the row-producing source set and runtime for future manifests."""
    return {
        "generator": "experiments/full_dataset.py",
        "source_sha256": _source_fingerprints(),
        "runtime": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    generate_dataset(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
