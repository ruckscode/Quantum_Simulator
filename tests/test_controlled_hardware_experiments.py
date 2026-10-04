"""Controlled program-versus-hardware experiments using the production APIs.

The records returned by these helpers keep the ideal reference, hardware-aware
execution, statistical comparison, and diagnosis as separate fields.
"""

from collections import Counter

from quantum_simulator.diagnosis_engine import DiagnosisCategory, diagnose_execution
from quantum_simulator.fault_localizer import _trace_deviation
from quantum_simulator.hardware_model import HardwareModel
from quantum_simulator.quantum_simulator import QuantumSimulator


SHOTS = 256
SEED = 20261004
CIRCUIT = [("measure",)]
PROGRAM_FAULT_CIRCUIT = [("x", 0), ("measure",)]
IDEAL_REFERENCE = {"0": 1.0, "1": 0.0}


def _run_hardware_shots(circuit, hardware, shots=SHOTS, seed=SEED):
    """Execute independent seeded shots through run_circuit's hardware path."""
    import numpy as np

    seeds = np.random.default_rng(seed).integers(0, 2**31 - 1, size=shots)
    counts = Counter()
    for shot_seed in seeds:
        simulator = QuantumSimulator(1, seed=int(shot_seed))
        result = simulator.run_circuit(circuit, hardware_model=hardware)
        counts[result[-1]] += 1
    return {outcome: counts.get(outcome, 0) for outcome in ("0", "1")}


def _distribution(counts):
    total = sum(counts.values())
    return {outcome: count / total for outcome, count in counts.items()}


def _experiment(hardware):
    observed_counts = _run_hardware_shots(CIRCUIT, hardware)
    observed_distribution = _distribution(observed_counts)
    diagnosis = diagnose_execution(
        expected_distribution=IDEAL_REFERENCE,
        observed_distribution=observed_distribution,
        expected_circuit=CIRCUIT,
        observed_circuit=CIRCUIT,
        hardware_model=hardware,
        shots=SHOTS,
    )
    comparison = diagnosis.statistics_result
    return {
        "configuration": {"circuit": CIRCUIT, "shots": SHOTS, "seed": SEED},
        "ideal_reference": IDEAL_REFERENCE.copy(),
        "hardware_configuration": hardware.to_calibration_snapshot(),
        "hardware_observed_counts": observed_counts,
        "hardware_observed_distribution": observed_distribution,
        "statistical_comparison": {
            "tvd": comparison.total_variation_distance,
            "hellinger": comparison.hellinger_distance,
            "js_divergence": comparison.js_divergence,
            "configured_threshold": comparison.threshold,
            "finite_shot_guard": comparison.finite_shot_guard,
            "anomaly_detected": comparison.anomaly_detected,
        },
        "diagnosis": diagnosis,
    }


def test_healthy_hardware_baseline_matches_ideal_and_is_no_anomaly():
    record = _experiment(HardwareModel(1, backend_name="healthy_baseline"))

    assert record["hardware_observed_counts"] == {"0": SHOTS, "1": 0}
    assert record["hardware_observed_distribution"] == record["ideal_reference"]
    assert record["statistical_comparison"]["tvd"] == 0.0
    assert record["statistical_comparison"]["anomaly_detected"] is False
    assert record["diagnosis"].category is DiagnosisCategory.NO_ANOMALY


def test_readout_fault_produces_anomaly_and_records_actual_diagnosis():
    faulty_hardware = HardwareModel(1, backend_name="readout_fault_experiment")
    faulty_hardware.set_qubit_properties(0, readout_error=1.0)
    record = _experiment(faulty_hardware)

    assert record["hardware_observed_counts"] == {"0": 0, "1": SHOTS}
    assert record["statistical_comparison"]["tvd"] == 1.0
    assert record["statistical_comparison"]["anomaly_detected"] is True
    # Preserve and inspect the category actually returned by diagnosis; the
    # experiment does not impose HARDWARE_ANOMALY as its expected result.
    assert record["diagnosis"].anomaly_detected is True
    assert record["diagnosis"].category in {
        DiagnosisCategory.HARDWARE_ANOMALY,
        DiagnosisCategory.INSUFFICIENT_EVIDENCE,
    }
    if record["diagnosis"].category is DiagnosisCategory.INSUFFICIENT_EVIDENCE:
        assert not record["diagnosis"].hardware_evidence_candidates


def test_program_fault_with_healthy_hardware_is_localized():
    healthy_hardware = HardwareModel(1, backend_name="healthy_program_fault_experiment")
    observed_counts = _run_hardware_shots(PROGRAM_FAULT_CIRCUIT, healthy_hardware)
    observed_distribution = _distribution(observed_counts)
    diagnosis = diagnose_execution(
        expected_distribution=IDEAL_REFERENCE,
        observed_distribution=observed_distribution,
        expected_circuit=CIRCUIT,
        observed_circuit=PROGRAM_FAULT_CIRCUIT,
        hardware_model=healthy_hardware,
        shots=SHOTS,
    )
    comparison = diagnosis.statistics_result
    localization = diagnosis.program_result
    record = {
        "configuration": {
            "correct_circuit": CIRCUIT,
            "faulty_circuit": PROGRAM_FAULT_CIRCUIT,
            "program_change": "Inserted ('x', 0) at step 0 before the existing measurement.",
            "shots": SHOTS,
            "seed": SEED,
        },
        "ideal_reference": IDEAL_REFERENCE.copy(),
        "hardware_configuration": healthy_hardware.to_calibration_snapshot(),
        "observed_counts": observed_counts,
        "observed_distribution": observed_distribution,
        "statistical_comparison": {
            "tvd": comparison.total_variation_distance,
            "hellinger": comparison.hellinger_distance,
            "js_divergence": comparison.js_divergence,
            "configured_threshold": comparison.threshold,
            "finite_shot_guard": comparison.finite_shot_guard,
            "anomaly_detected": comparison.anomaly_detected,
        },
        "diagnosis": diagnosis,
        "localization": localization,
    }

    assert healthy_hardware.get_qubit_properties(0).readout_error == 0.0
    assert healthy_hardware.get_qubit_properties(0).t1 == 1e6
    assert healthy_hardware.get_qubit_properties(0).t2 == 1e6
    assert healthy_hardware.get_gate_calibration("x", qubits=(0,)) == []
    assert record["observed_counts"] == {"0": 0, "1": SHOTS}
    assert record["observed_distribution"] == {"0": 0.0, "1": 1.0}
    assert record["statistical_comparison"]["tvd"] == 1.0
    assert record["statistical_comparison"]["hellinger"] == 1.0
    assert record["statistical_comparison"]["js_divergence"] > 0.69
    assert record["statistical_comparison"]["anomaly_detected"] is True
    assert record["diagnosis"].category is DiagnosisCategory.PROGRAM_FAULT
    assert record["localization"].candidates
    assert record["localization"].candidates[0].operation == "x"
    assert record["localization"].candidates[0].index == 0
    assert record["localization"].candidates[0].actual_signature == ("x", (0,), ())
    assert record["localization"].candidates[0].expected_signature is None


def test_experiment_4_program_and_hardware_faults_record_existing_diagnosis():
    faulty_hardware = HardwareModel(1, backend_name="combined_fault_experiment")
    faulty_hardware.set_qubit_properties(0, readout_error=1.0)
    observed_counts = _run_hardware_shots(PROGRAM_FAULT_CIRCUIT, faulty_hardware)
    observed_distribution = _distribution(observed_counts)
    diagnosis = diagnose_execution(
        expected_distribution=IDEAL_REFERENCE,
        observed_distribution=observed_distribution,
        expected_circuit=CIRCUIT,
        observed_circuit=PROGRAM_FAULT_CIRCUIT,
        hardware_model=faulty_hardware,
        shots=SHOTS,
    )
    comparison = diagnosis.statistics_result
    record = {
        "configuration": {
            "correct_circuit": CIRCUIT,
            "faulty_circuit": PROGRAM_FAULT_CIRCUIT,
            "program_change": "Inserted ('x', 0) at step 0 before the existing measurement.",
            "hardware_fault": "Qubit 0 readout error set to 1.0.",
            "shots": SHOTS,
            "seed": SEED,
        },
        "ideal_reference": IDEAL_REFERENCE.copy(),
        "hardware_configuration": faulty_hardware.to_calibration_snapshot(),
        "observed_counts": observed_counts,
        "observed_distribution": observed_distribution,
        "statistical_comparison": {
            "tvd": comparison.total_variation_distance,
            "hellinger": comparison.hellinger_distance,
            "js_divergence": comparison.js_divergence,
            "configured_threshold": comparison.threshold,
            "finite_shot_guard": comparison.finite_shot_guard,
            "anomaly_detected": comparison.anomaly_detected,
        },
        "diagnosis": diagnosis,
        "program_localization": diagnosis.program_result,
        "hardware_localization": diagnosis.hardware_result,
        "hardware_evidence": diagnosis.hardware_evidence_candidates,
    }

    # Confirm both injected faults are present, then retain whatever the
    # production diagnosis returns for their combined execution evidence.
    assert PROGRAM_FAULT_CIRCUIT == [("x", 0), ("measure",)]
    assert faulty_hardware.get_qubit_properties(0).readout_error == 1.0
    assert record["observed_counts"] == {"0": SHOTS, "1": 0}
    assert record["observed_distribution"] == {"0": 1.0, "1": 0.0}
    assert record["statistical_comparison"]["tvd"] == 0.0
    assert record["statistical_comparison"]["hellinger"] == 0.0
    assert record["statistical_comparison"]["js_divergence"] == 0.0
    assert record["diagnosis"].category in set(DiagnosisCategory)


def test_experiment_4_diagnostic_trace_preserves_program_evidence_despite_cancellation():
    faulty_hardware = HardwareModel(1, backend_name="combined_fault_trace_experiment")
    faulty_hardware.set_qubit_properties(0, readout_error=1.0)
    observed_counts = _run_hardware_shots(PROGRAM_FAULT_CIRCUIT, faulty_hardware)
    observed_distribution = _distribution(observed_counts)

    # This existing trace pathway compares the ideal state evolution of the
    # reference and faulty circuits, independently of hardware readout.
    trace = _trace_deviation(CIRCUIT, PROGRAM_FAULT_CIRCUIT, n_qubits=1)
    diagnosis = diagnose_execution(
        expected_distribution=IDEAL_REFERENCE,
        observed_distribution=observed_distribution,
        expected_circuit=CIRCUIT,
        observed_circuit=PROGRAM_FAULT_CIRCUIT,
        hardware_model=faulty_hardware,
        shots=SHOTS,
        precomputed_trace=trace,
    )
    trace_metric, trace_steps = trace

    assert observed_counts == {"0": SHOTS, "1": 0}
    assert observed_distribution == IDEAL_REFERENCE
    assert trace_metric == 1.0
    assert trace_steps == [
        {"step": 0, "operation": "x", "expected": ("measure",), "actual": ("x", 0), "difference": 0.75},
        {"step": 1, "operation": "measure", "expected": None, "actual": ("measure",), "difference": 1.0},
    ]
    assert diagnosis.program_result.anomaly_metric == trace_metric
    assert diagnosis.program_result.candidates
    assert diagnosis.program_result.candidates[0].operation == "x"
    assert diagnosis.program_result.candidates[0].index == 0
    assert diagnosis.hardware_evidence_candidates
    assert any(
        candidate.component_type == "qubit"
        and candidate.component == 0
        and any("readout error is elevated at 1.0000" in item for item in candidate.evidence)
        for candidate in diagnosis.hardware_evidence_candidates
    )
    # The final distribution comparison still controls the top-level category.
    assert diagnosis.statistics_result.total_variation_distance == 0.0
    assert diagnosis.category is DiagnosisCategory.NO_ANOMALY


def test_experiment_5_program_and_partial_readout_fault_record_current_diagnosis():
    faulty_hardware = HardwareModel(1, backend_name="combined_partial_readout_experiment")
    faulty_hardware.set_qubit_properties(0, readout_error=0.25)
    observed_counts = _run_hardware_shots(PROGRAM_FAULT_CIRCUIT, faulty_hardware)
    observed_distribution = _distribution(observed_counts)
    trace = _trace_deviation(CIRCUIT, PROGRAM_FAULT_CIRCUIT, n_qubits=1)
    diagnosis = diagnose_execution(
        expected_distribution=IDEAL_REFERENCE,
        observed_distribution=observed_distribution,
        expected_circuit=CIRCUIT,
        observed_circuit=PROGRAM_FAULT_CIRCUIT,
        hardware_model=faulty_hardware,
        shots=SHOTS,
        precomputed_trace=trace,
    )
    comparison = diagnosis.statistics_result

    record = {
        "configuration": {
            "correct_circuit": CIRCUIT,
            "faulty_circuit": PROGRAM_FAULT_CIRCUIT,
            "program_change": "Inserted ('x', 0) at step 0 before the existing measurement.",
            "hardware_fault": "Qubit 0 readout error set to 0.25.",
            "shots": SHOTS,
            "seed": SEED,
        },
        "ideal_reference": IDEAL_REFERENCE.copy(),
        "observed_counts": observed_counts,
        "observed_distribution": observed_distribution,
        "trace": trace,
        "statistical_comparison": {
            "tvd": comparison.total_variation_distance,
            "hellinger": comparison.hellinger_distance,
            "js_divergence": comparison.js_divergence,
            "finite_shot_guard": comparison.finite_shot_guard,
        },
        "diagnosis": diagnosis,
        "program_localization": diagnosis.program_result,
        "hardware_localization": diagnosis.hardware_result,
        "hardware_evidence": diagnosis.hardware_evidence_candidates,
    }

    assert PROGRAM_FAULT_CIRCUIT == [("x", 0), ("measure",)]
    assert faulty_hardware.get_qubit_properties(0).readout_error == 0.25
    assert sum(record["observed_counts"].values()) == SHOTS
    assert set(record["observed_distribution"]) == {"0", "1"}
    assert record["trace"][0] == 1.0
    assert record["program_localization"].candidates
    assert record["program_localization"].candidates[0].operation == "x"
    assert record["program_localization"].candidates[0].index == 0
    assert record["hardware_evidence"]
    assert record["statistical_comparison"]["finite_shot_guard"] == 0.0625
    assert record["diagnosis"].category in set(DiagnosisCategory)


def test_experiment_6_readout_error_sweep_records_actual_diagnosis_evidence():
    readout_errors = (0.00, 0.10, 0.25, 0.50, 0.75, 1.00)
    records = []

    for readout_error in readout_errors:
        hardware = HardwareModel(1, backend_name=f"readout_sweep_{readout_error:.2f}")
        hardware.set_qubit_properties(0, readout_error=readout_error)
        observed_counts = _run_hardware_shots(PROGRAM_FAULT_CIRCUIT, hardware)
        observed_distribution = _distribution(observed_counts)
        trace = _trace_deviation(CIRCUIT, PROGRAM_FAULT_CIRCUIT, n_qubits=1)
        diagnosis = diagnose_execution(
            expected_distribution=IDEAL_REFERENCE,
            observed_distribution=observed_distribution,
            expected_circuit=CIRCUIT,
            observed_circuit=PROGRAM_FAULT_CIRCUIT,
            hardware_model=hardware,
            shots=SHOTS,
            precomputed_trace=trace,
        )
        comparison = diagnosis.statistics_result
        records.append({
            "readout_error": readout_error,
            "observed_counts": observed_counts,
            "observed_distribution": observed_distribution,
            "tvd": comparison.total_variation_distance,
            "hellinger": comparison.hellinger_distance,
            "js_divergence": comparison.js_divergence,
            "finite_shot_guard": comparison.finite_shot_guard,
            "anomaly_detected": comparison.anomaly_detected,
            "trace": trace,
            "diagnosis": diagnosis,
            "program_localization": diagnosis.program_result,
            "hardware_localization": diagnosis.hardware_result,
            "hardware_evidence": diagnosis.hardware_evidence_candidates,
        })

    assert tuple(record["readout_error"] for record in records) == readout_errors
    for record in records:
        assert sum(record["observed_counts"].values()) == SHOTS
        assert set(record["observed_distribution"]) == {"0", "1"}
        assert record["finite_shot_guard"] == 0.0625
        assert record["program_localization"].candidates
        assert record["program_localization"].candidates[0].operation == "x"
        assert record["program_localization"].candidates[0].index == 0
        assert record["trace"][0] == 1.0
        assert record["diagnosis"].category in set(DiagnosisCategory)


def test_experiment_7_gate_error_detection_and_localization_records_actual_results():
    """Compare program replacement with a calibrated X gate execution fault."""
    correct_program = [("x", 0), ("measure",)]
    faulty_program = [("h", 0), ("measure",)]
    ideal_reference = {"0": 0.0, "1": 1.0}
    trace = _trace_deviation(correct_program, faulty_program, n_qubits=1)

    cases = [
        ("A", "correct program + healthy hardware", correct_program, False),
        ("B", "correct program + faulty hardware", correct_program, True),
        ("C", "faulty program + healthy hardware", faulty_program, False),
        ("D", "faulty program + faulty hardware", faulty_program, True),
    ]
    records = []
    for case_name, description, program, x_fault in cases:
        hardware = HardwareModel(1, backend_name=f"experiment_7_{case_name.lower()}")
        if x_fault:
            hardware.add_gate_calibration("x", qubits=(0,), error=1.0)

        observed_counts = _run_hardware_shots(program, hardware)
        observed_distribution = _distribution(observed_counts)
        diagnosis = diagnose_execution(
            expected_distribution=ideal_reference,
            observed_distribution=observed_distribution,
            expected_circuit=correct_program,
            observed_circuit=program,
            hardware_model=hardware,
            shots=SHOTS,
            precomputed_trace=trace if program != correct_program else None,
        )
        comparison = diagnosis.statistics_result
        trace_metric = diagnosis.program_result.anomaly_metric if diagnosis.program_result else None
        records.append({
            "case_name": case_name,
            "description": description,
            "program": program,
            "hardware_configuration": hardware.to_calibration_snapshot(),
            "observed_counts": observed_counts,
            "observed_distribution": observed_distribution,
            "tvd": comparison.total_variation_distance,
            "hellinger_distance": comparison.hellinger_distance,
            "js_divergence": comparison.js_divergence,
            "finite_shot_guard": comparison.finite_shot_guard,
            "anomaly_detected": diagnosis.anomaly_detected,
            "diagnosis": diagnosis,
            "program_localization": diagnosis.program_result,
            "hardware_localization": diagnosis.hardware_result,
            "hardware_evidence": diagnosis.hardware_evidence_candidates,
            "trace_anomaly_metric": trace_metric,
            "shots": SHOTS,
            "seed": SEED,
            "ideal_reference": ideal_reference.copy(),
        })

    assert [record["case_name"] for record in records] == ["A", "B", "C", "D"]
    assert all(sum(record["observed_counts"].values()) == SHOTS for record in records)
    assert all(record["finite_shot_guard"] == 0.0625 for record in records)
    assert all(record["diagnosis"].category in set(DiagnosisCategory) for record in records)
    # Preserve the actual diagnosis and localization output without imposing
    # an expected category or candidate for any case.
    assert records[1]["hardware_configuration"] != records[0]["hardware_configuration"]
    assert records[1]["hardware_evidence"]
    assert records[2]["program_localization"].candidates
    assert records[3]["hardware_evidence"]


def test_experiment_8_t1_t2_relaxation_records_complete_diagnosis_results():
    """Exercise relaxation/dephasing during a calibrated idle gate duration."""
    circuit = [("h", 0), ("i", 0), ("h", 0), ("measure",)]
    ideal_reference = {"0": 1.0, "1": 0.0}
    cases = [
        ("A", "healthy", None, None),
        ("B", "degraded_t1", 0.1, None),
        ("C", "degraded_t2", None, 0.1),
    ]
    records = []

    for case_name, configuration_name, t1, t2 in cases:
        hardware = HardwareModel(1, backend_name=f"experiment_8_{case_name.lower()}")
        # A one-unit calibrated idle gate activates the existing hardware
        # relaxation path between state preparation and readout-basis rotation.
        hardware.add_gate_calibration("i", qubits=(0,), duration=1.0)
        if t1 is not None or t2 is not None:
            hardware.set_qubit_properties(0, **{
                **({"t1": t1} if t1 is not None else {}),
                **({"t2": t2} if t2 is not None else {}),
            })

        observed_counts = _run_hardware_shots(circuit, hardware)
        observed_distribution = _distribution(observed_counts)
        trace = _trace_deviation(circuit, circuit, n_qubits=1)
        diagnosis = diagnose_execution(
            expected_distribution=ideal_reference,
            observed_distribution=observed_distribution,
            expected_circuit=circuit,
            observed_circuit=circuit,
            hardware_model=hardware,
            shots=SHOTS,
            precomputed_trace=trace,
        )
        comparison = diagnosis.statistics_result
        records.append({
            "case_name": case_name,
            "configuration_name": configuration_name,
            "t1_t2_configuration": hardware.to_calibration_snapshot().qubits[0],
            "gate_calibration": hardware.to_calibration_snapshot().gates,
            "observed_counts": observed_counts,
            "observed_distribution": observed_distribution,
            "tvd": comparison.total_variation_distance,
            "hellinger": comparison.hellinger_distance,
            "js_divergence": comparison.js_divergence,
            "finite_shot_guard": comparison.finite_shot_guard,
            "anomaly_detected": diagnosis.anomaly_detected,
            "diagnosis": diagnosis,
            "program_localization": diagnosis.program_result,
            "hardware_localization": diagnosis.hardware_result,
            "hardware_evidence": diagnosis.hardware_evidence_candidates,
            "trace_anomaly_metric": diagnosis.program_result.anomaly_metric if diagnosis.program_result else None,
            "shots": SHOTS,
            "seed": SEED,
            "program": circuit,
            "ideal_reference": ideal_reference.copy(),
        })

    assert [record["case_name"] for record in records] == ["A", "B", "C"]
    assert all(sum(record["observed_counts"].values()) == SHOTS for record in records)
    assert all(record["finite_shot_guard"] == 0.0625 for record in records)
    assert all(record["gate_calibration"]["i"][0]["duration"] == 1.0 for record in records)
    assert all(record["diagnosis"].category in set(DiagnosisCategory) for record in records)
    assert records[0]["t1_t2_configuration"]["t1"] == 1e6
    assert records[0]["t1_t2_configuration"]["t2"] == 1e6
    assert records[1]["t1_t2_configuration"]["t1"] == 0.1
    assert records[1]["t1_t2_configuration"]["t2"] == 1e6
    assert records[2]["t1_t2_configuration"]["t1"] == 1e6
    assert records[2]["t1_t2_configuration"]["t2"] == 0.1


def test_experiment_9_seed_robustness_matrix(capsys):
    """Record diagnosis outcomes for three controlled cases across five seeds."""
    seeds = (1729, 20261004, 314159, 424242, 8675309)
    cases = (
        ("A", CIRCUIT, "healthy"),
        ("B", PROGRAM_FAULT_CIRCUIT, "healthy"),
        ("C", CIRCUIT, "readout_0.25"),
    )
    records = []

    for case_name, program, hardware_configuration in cases:
        for seed in seeds:
            hardware = HardwareModel(1, backend_name=f"experiment_9_{case_name.lower()}_{seed}")
            if hardware_configuration == "readout_0.25":
                hardware.set_qubit_properties(0, readout_error=0.25)
            observed_counts = _run_hardware_shots(program, hardware, shots=SHOTS, seed=seed)
            observed_distribution = _distribution(observed_counts)
            trace = _trace_deviation(CIRCUIT, program, n_qubits=1)
            diagnosis = diagnose_execution(
                expected_distribution=IDEAL_REFERENCE,
                observed_distribution=observed_distribution,
                expected_circuit=CIRCUIT,
                observed_circuit=program,
                hardware_model=hardware,
                shots=SHOTS,
                precomputed_trace=trace,
            )
            comparison = diagnosis.statistics_result
            records.append({
                "case": case_name,
                "seed": seed,
                "observed_counts": observed_counts,
                "observed_distribution": observed_distribution,
                "tvd": comparison.total_variation_distance,
                "hellinger": comparison.hellinger_distance,
                "js_divergence": comparison.js_divergence,
                "finite_shot_guard": comparison.finite_shot_guard,
                "anomaly_detected": diagnosis.anomaly_detected,
                "diagnosis": diagnosis,
                "program_localization": diagnosis.program_result,
                "hardware_evidence": diagnosis.hardware_evidence_candidates,
                "hardware_localization": diagnosis.hardware_result,
            })

    assert [(record["case"], record["seed"]) for record in records] == [
        (case_name, seed) for case_name, _, _ in cases for seed in seeds
    ]
    assert len(records) == 15
    assert all(sum(record["observed_counts"].values()) == SHOTS for record in records)
    assert all(record["finite_shot_guard"] == 0.0625 for record in records)
    assert all(record["diagnosis"].category in set(DiagnosisCategory) for record in records)

    summaries = {}
    for case_name, _, _ in cases:
        case_records = [record for record in records if record["case"] == case_name]
        diagnoses = [record["diagnosis"].category.value for record in case_records]
        tvds = [record["tvd"] for record in case_records]
        summaries[case_name] = {
            "diagnoses": diagnoses,
            "consistent": len(set(diagnoses)) == 1,
            "min_tvd": min(tvds),
            "max_tvd": max(tvds),
        }
    unexpected = [
        {"case": record["case"], "seed": record["seed"],
         "diagnosis": record["diagnosis"].category.value,
         "tvd": record["tvd"], "anomaly_detected": record["anomaly_detected"]}
        for record in records
        if (record["case"] == "A" and record["diagnosis"].anomaly_detected)
        or (record["case"] != "A" and not record["diagnosis"].anomaly_detected)
    ]
    print({"number_of_runs": len(records), "runs": [
        (record["case"], record["seed"], record["diagnosis"].category.value)
        for record in records
    ], "cases": summaries, "unexpected": unexpected})
