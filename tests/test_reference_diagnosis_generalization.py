"""Held-out evaluation on structures absent from the reference case corpus."""

import json
import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.robustness_evaluation import _family_circuit, _program_fault
from quantum_simulator.diagnosis_engine import DiagnosisCategory, diagnose_circuit
from quantum_simulator.hardware_model import HardwareModel
from quantum_simulator.quantum_simulator import QuantumSimulator


SPEC_PATH = Path(__file__).parent / "fixtures" / "reference_diagnosis_generalization_cases.json"
CONTRACT_PATH = Path(__file__).parent / "fixtures" / "reference_diagnosis_acceptance_v2.json"
ABSTAIN_CATEGORIES = {DiagnosisCategory.AMBIGUOUS, DiagnosisCategory.INSUFFICIENT_EVIDENCE}


def _as_instructions(operations):
    return [tuple(item) if isinstance(item, list) else item for item in operations]


def _mutate_equivalently(reference, mutation):
    observed = list(reference)
    kind = mutation["kind"]
    if kind == "insert_identity":
        measure_index = next(index for index, operation in enumerate(observed) if operation[0] == "measure")
        observed.insert(measure_index, ("i", mutation["qubit"]))
    elif kind == "reorder_disjoint_x_and_ccx":
        observed[2], observed[3] = observed[3], observed[2]
    elif kind == "reverse_commuting_cz_ring":
        indices = [index for index, operation in enumerate(observed) if operation[0] == "cz"]
        gates = [observed[index] for index in indices]
        for index, operation in zip(indices, reversed(gates)):
            observed[index] = operation
    else:
        raise ValueError(f"unknown equivalent mutation: {kind}")
    return observed


def build_heldout_cases():
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    cases = []
    for family in spec["families"]:
        if "generator" in family:
            generator = family["generator"]
            references = {
                seed: _family_circuit(
                    generator["family"], generator["n_qubits"], generator["depth"], seed,
                )
                for seed in spec["seeds"]
            }
        else:
            reference = _as_instructions(family["reference"])
            references = {seed: reference for seed in spec["seeds"]}

        mutations = {}
        for seed, reference in references.items():
            mutated, _generator_metadata = _program_fault(reference, family["fault_injection"])
            mutations[seed] = mutated

        for seed, reference in references.items():
            mutated = mutations[seed]
            equivalent = _mutate_equivalently(reference, family["equivalent_mutation"])
            hardware_error = family.get("hardware_error", 1.0)
            cases.extend([
                {
                    "id": f"{family['name']}_correct_{seed}", "family": family["name"],
                    "scenario": "correct", "reference": reference, "submitted": reference,
                    "hardware": None, "seed": seed, "shots": spec["shots"],
                    "expected": spec["expected_by_scenario"]["correct"],
                },
                {
                    "id": f"{family['name']}_equivalent_{seed}", "family": family["name"],
                    "scenario": "equivalent", "reference": reference, "submitted": equivalent,
                    "hardware": None, "seed": seed, "shots": spec["shots"],
                    "expected": spec["expected_by_scenario"]["equivalent"],
                },
                {
                    "id": f"{family['name']}_program_fault_{seed}", "family": family["name"],
                    "scenario": "program_fault", "reference": reference, "submitted": mutated,
                    "fault_target": family["expected_fault_target"], "hardware": None, "seed": seed,
                    "shots": spec["shots"], "expected": spec["expected_by_scenario"]["program_fault"],
                },
                {
                    "id": f"{family['name']}_hardware_{seed}", "family": family["name"],
                    "scenario": "hardware_anomaly", "reference": reference, "submitted": reference,
                    "hardware": {"n_qubits": max(q for op in reference for q in _operation_qubits(op)) + 1,
                                 "qubit": family["hardware_qubit"], "readout_error": hardware_error},
                    "hardware_target": ("qubit", family["hardware_qubit"]),
                    "seed": seed, "shots": spec["shots"],
                    "expected": spec["expected_by_scenario"]["hardware_anomaly"],
                },
                {
                    "id": f"{family['name']}_ambiguous_{seed}", "family": family["name"],
                    "scenario": "ambiguous", "reference": reference, "submitted": mutated,
                    "fault_target": family["expected_fault_target"],
                    "hardware": {"n_qubits": max(q for op in reference for q in _operation_qubits(op)) + 1,
                                 "qubit": family["hardware_qubit"], "readout_error": 0.45},
                    "hardware_target": ("qubit", family["hardware_qubit"]),
                    "seed": seed, "shots": spec["shots"],
                    "expected": spec["expected_by_scenario"]["ambiguous"],
                },
            ])
    return cases


def _operation_qubits(operation):
    name = operation[0]
    if name == "measure":
        operands = operation[1] if len(operation) > 1 else ()
        return tuple(operands) if isinstance(operands, (list, tuple)) else (operands,)
    if name in {"cx", "cnot", "cz", "cp", "crz", "swap"}:
        return tuple(operation[1:3])
    if name in {"ccx", "toffoli"}:
        return tuple(operation[1:4])
    return tuple(operation[1:2])


def evaluate_heldout_cases():
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    overrides = contract["outcome_overrides"]
    rows = []
    for case in build_heldout_cases():
        hardware = None
        if case["hardware"]:
            profile = case["hardware"]
            hardware = HardwareModel(profile["n_qubits"])
            hardware.set_qubit_properties(profile["qubit"], readout_error=profile["readout_error"])
        # Ground truth, scenario name, and mutation metadata are deliberately
        # excluded from this public API call.
        result = diagnose_circuit(
            case["submitted"], reference_circuit=case["reference"],
            hardware_model=hardware, shots=case["shots"], seed=case["seed"],
        )
        observed = {
            "detection": result.statistics_result.anomaly_detected,
            "classification": result.category.value,
            "source_localized": bool(result.program_evidence_candidates),
            "hardware_localized": bool(result.hardware_evidence_candidates),
            "abstain": result.category in ABSTAIN_CATEGORIES,
        }
        source_candidates = [
            (candidate.index, candidate.operation, tuple(candidate.qubits))
            for candidate in result.program_evidence_candidates
        ]
        hardware_candidates = [
            (candidate.component_type, tuple(candidate.component)
             if isinstance(candidate.component, (tuple, list)) else candidate.component)
            for candidate in result.hardware_evidence_candidates
        ]
        source_target = case.get("fault_target")
        expected_source = None if source_target is None else (
            source_target["index"], source_target["operation"], tuple(source_target["qubits"]),
        )
        hardware_target = case.get("hardware_target")
        rows.append({
            "id": case["id"], "family": case["family"], "scenario": case["scenario"],
            "expected": overrides.get(case["id"], case["expected"]), "observed": observed,
            "expected_source_target": expected_source, "source_candidates": source_candidates,
            "expected_hardware_target": hardware_target, "hardware_candidates": hardware_candidates,
            "source_target_hit": expected_source is not None and expected_source in source_candidates,
            "hardware_target_hit": hardware_target is not None and hardware_target in hardware_candidates,
        })
    return rows


def test_heldout_generalization_outcomes_and_targets():
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    rows = evaluate_heldout_cases()
    assert len(rows) == 45
    assert {row["family"] for row in rows} == {family["name"] for family in spec["families"]}
    assert {case["seed"] for case in build_heldout_cases()} == set(spec["seeds"])
    mismatches = []
    for row in rows:
        if row["observed"] != row["expected"]:
            mismatches.append(f"{row['id']}: expected {row['expected']}, observed {row['observed']}")
        if row["expected_source_target"] is not None:
            if not row["source_target_hit"]:
                mismatches.append(f"{row['id']}: expected source target was not localized")
        if row["expected_hardware_target"] is not None:
            if not row["hardware_target_hit"]:
                mismatches.append(f"{row['id']}: expected hardware target was not localized")
    assert not mismatches, "held-out evaluation mismatches:\n" + "\n".join(mismatches)


def test_v2_contract_is_versioned_and_preserves_v1_fixture_bytes():
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert contract["version"] == "2.0.0"
    assert contract["status"] == "active"
    assert contract["detector"] == {
        "method": "conditional two-sample TVD permutation test",
        "permutations": 999,
        "alpha": 0.05,
        "p_value_correction": "add one",
        "minimum_observed_tvd": 0.05,
        "seed": 0,
    }
    for entry in contract["historical_fixtures"].values():
        source = SPEC_PATH.parent / entry["path"]
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        assert digest == entry["sha256"]


def test_frozen_heldout_mismatches_are_explained_by_distribution_resolution():
    """Retain exact-effect evidence while exercising calibrated count testing."""
    selected = {
        "qft_like_3q_program_fault_11": ("subthreshold", 0.01953125),
        "qft_like_3q_program_fault_29": ("subthreshold", 0.0),
        "qft_like_3q_program_fault_47": ("subthreshold", 0.01953125),
        "ring_cluster_4q_correct_11": ("control_false_positive", 0.0859375),
        "ring_cluster_4q_correct_29": ("control_false_positive", 0.125),
        "ring_cluster_4q_equivalent_11": ("control_false_positive", 0.0859375),
        "ring_cluster_4q_equivalent_29": ("control_false_positive", 0.125),
    }

    def exact_distribution(operations, n_qubits):
        simulator = QuantumSimulator(n_qubits)
        simulator.run_circuit(
            [operation for operation in operations if operation[0] != "measure"]
        )
        return {
            format(index, f"0{n_qubits}b"): float(abs(amplitude) ** 2)
            for index, amplitude in enumerate(simulator.state)
        }

    seen = set()
    for case in build_heldout_cases():
        if case["id"] not in selected:
            continue
        seen.add(case["id"])
        expected_profile, sampled_tvd = selected[case["id"]]
        n_qubits = max(
            qubit
            for operation in case["reference"]
            for qubit in _operation_qubits(operation)
        ) + 1
        reference = exact_distribution(case["reference"], n_qubits)
        submitted = exact_distribution(case["submitted"], n_qubits)
        exact_tvd = 0.5 * sum(
            abs(reference.get(key, 0.0) - submitted.get(key, 0.0))
            for key in reference.keys() | submitted.keys()
        )

        hardware = None
        if case["hardware"]:
            profile = case["hardware"]
            hardware = HardwareModel(profile["n_qubits"])
            hardware.set_qubit_properties(
                profile["qubit"], readout_error=profile["readout_error"]
            )
        result = diagnose_circuit(
            case["submitted"], reference_circuit=case["reference"],
            hardware_model=hardware, shots=case["shots"], seed=case["seed"],
        )
        statistics = result.statistics_result
        assert statistics is not None
        assert statistics.total_variation_distance == pytest.approx(sampled_tvd)
        assert statistics.method == "conditional_tvd_permutation"
        assert statistics.p_value is not None

        if expected_profile == "subthreshold":
            assert 0.0 < exact_tvd < statistics.threshold
        else:
            assert exact_tvd == pytest.approx(0.0)
    assert seen == set(selected)
