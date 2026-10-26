"""Independent, explicitly labeled validation cases for the reference API."""

import json
from pathlib import Path

from quantum_simulator.diagnosis_engine import DiagnosisCategory, diagnose_circuit
from quantum_simulator.hardware_model import HardwareModel
from quantum_simulator.quantum_simulator import QuantumSimulator


CASES_PATH = Path(__file__).parent / "fixtures" / "reference_diagnosis_validation_cases.json"
CONTRACT_PATH = Path(__file__).parent / "fixtures" / "reference_diagnosis_acceptance_v2.json"
ABSTAIN_CATEGORIES = {
    DiagnosisCategory.AMBIGUOUS,
    DiagnosisCategory.INSUFFICIENT_EVIDENCE,
}


def expected_multiqubit_targets(case):
    """Return predeclared targets from the case definition, never model output."""
    family = case.get("family")
    group = case.get("group")
    source = []
    hardware = []
    if group == "behaviorally_equivalent_structural_multi_qubit":
        source = {
            "bell_2q": [(0, "h", (1,)), (1, "cx", (1, 0))],
            "ghz_3q": [(0, "h", (2,)), (1, "cx", (2, 1)), (2, "cx", (1, 0))],
            "parity_3q": [(0, "h", (1,)), (1, "h", (0,))],
        }[family]
    elif group in {"program_fault_multi_qubit", "ambiguous_multi_qubit"}:
        source = {
            "bell_2q": [(1, "cx", (0, 1))],
            "ghz_3q": [(2, "cx", (1, 2))],
            "parity_3q": [(2, "cx", (0, 2))],
        }[family]
    elif group == "measurement_order_multi_qubit":
        source = [(1, "measure", (1, 0))]

    if group in {"hardware_model_anomaly_multi_qubit", "ambiguous_multi_qubit"}:
        hardware = [("qubit", 0)]
    return {"source": source, "hardware": hardware}


def evaluate_validation_set():
    """Evaluate cases with labels applied only after diagnosis has returned."""
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    overrides = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))["outcome_overrides"]
    rows = []
    for case in cases:
        hardware = None
        if "hardware" in case:
            hardware = HardwareModel(case["hardware"]["n_qubits"])
            hardware.set_qubit_properties(
                0, readout_error=case["hardware"]["readout_error"],
            )
        # Only programs and simulator parameters enter the diagnosis API.
        # The group and expected outcome are read after the call for scoring.
        try:
            result = diagnose_circuit(
                case["submitted"],
                reference_circuit=case["reference"],
                hardware_model=hardware,
                shots=case["shots"],
                seed=case["seed"],
            )
        except ValueError as error:
            rows.append({
                "id": case["id"],
                "expected": overrides.get(case["id"], case["expected"]),
                "observed": {"validation_error": str(error)},
            })
            continue
        observed = {
            "detection": result.statistics_result.anomaly_detected,
            "classification": result.category.value,
            "source_localized": bool(result.program_evidence_candidates),
            "hardware_localized": bool(result.hardware_evidence_candidates),
            "abstain": result.category in ABSTAIN_CATEGORIES,
        }
        localization = None
        if "family" in case:
            targets = expected_multiqubit_targets(case)
            source_found = [
                (candidate.index, candidate.operation, tuple(candidate.qubits))
                for candidate in result.program_evidence_candidates
            ]
            hardware_found = [
                (candidate.component_type, tuple(candidate.component)
                 if isinstance(candidate.component, (list, tuple)) else candidate.component)
                for candidate in result.hardware_evidence_candidates
            ]
            localization = {
                "expected_source": targets["source"],
                "observed_source": source_found,
                "source_targets_hit": sum(target in source_found for target in targets["source"]),
                "expected_hardware": targets["hardware"],
                "observed_hardware": hardware_found,
                "hardware_targets_hit": sum(target in hardware_found for target in targets["hardware"]),
            }
        rows.append({
            "id": case["id"],
            "expected": overrides.get(case["id"], case["expected"]),
            "observed": observed,
            "localization": localization,
        })
    return rows


def test_reference_diagnosis_held_out_validation_axes():
    rows = evaluate_validation_set()
    assert len(rows) == 39
    for row in rows:
        if "validation_error" in row["expected"]:
            assert row["expected"]["validation_error"] in row["observed"]["validation_error"]
        else:
            assert row["observed"] == row["expected"], row["id"]
        if row.get("localization") is not None:
            result = row["localization"]
            assert result["source_targets_hit"] == len(result["expected_source"]), row["id"]
            assert result["hardware_targets_hit"] == len(result["expected_hardware"]), row["id"]


def test_validation_fixture_has_complete_independent_multiqubit_seed_matrix():
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    assert len(cases) == 39
    assert len({case["id"] for case in cases}) == len(cases)
    assert all(case.get("group") and case.get("expected") for case in cases)

    multi = [case for case in cases if "family" in case]
    assert len(multi) == 32
    assert {case["family"] for case in multi} == {"bell_2q", "ghz_3q", "parity_3q"}
    assert {case["seed"] for case in multi} == {211, 223}
    required_groups = {
        "correct_multi_qubit",
        "behaviorally_equivalent_structural_multi_qubit",
        "program_fault_multi_qubit",
        "hardware_model_anomaly_multi_qubit",
        "ambiguous_multi_qubit",
    }
    for family in {case["family"] for case in multi}:
        for group in required_groups:
            matching = [case for case in multi if case["family"] == family and case["group"] == group]
            assert [case["seed"] for case in matching] == [211, 223]
    mapping_cases = [case for case in multi if case["family"] == "bell_2q" and case["group"] == "measurement_order_multi_qubit"]
    assert [case["seed"] for case in mapping_cases] == [211, 223]


def test_multi_qubit_measurement_order_maps_qubits_to_classical_bits_as_documented():
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    case = next(
        item for item in cases
        if item["id"] == "reversed_qubit_classical_mapping_changes_output_order"
    )
    reference = QuantumSimulator(2, seed=1).run_circuit(case["reference"])
    submitted = QuantumSimulator(2, seed=1).run_circuit(case["submitted"])
    assert reference[-1] == {0: 1, 1: 0}
    assert submitted[-1] == {0: 0, 1: 1}
